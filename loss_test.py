import tensorflow as tf

def soft_quadratic_weighted_kappa(y_true, y_pred_softmax, num_classes):
    """
    Tính QWK dựa trên xác suất (softmax output)

    Args:
        y_true: Tensor shape (batch_size,) chứa nhãn thực tế (integers)
        y_pred_softmax: Tensor shape (batch_size, num_classes), đầu ra sau softmax
        num_classes: Số lớp phân loại

    Returns:
        Scalar tensor: QWK score (giá trị càng cao càng tốt)
    """
    # Tạo ma trận trọng số |i - j|^2 / (num_classes - 1)^2
    weights = tf.constant([[float(i - j) ** 2 for j in range(num_classes)] for i in range(num_classes)])
    weights = weights / ((num_classes - 1) ** 2)

    # One-hot encode y_true
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, dtype=tf.float32)

    # Tính O (Observed): confusion matrix mềm
    O = tf.matmul(tf.transpose(y_pred_softmax), y_true_one_hot)

    # Tính E (Expected): outer product của margin distribution
    hist_r = tf.reduce_sum(y_pred_softmax, axis=0)
    hist_c = tf.reduce_sum(y_true_one_hot, axis=0)
    batch_size = tf.shape(y_true)[0]
    E = tf.tensordot(hist_r, hist_c, axes=0) / tf.cast(batch_size, tf.float32)

    # Tính tử và mẫu của QWK
    numerator = tf.reduce_sum(weights * O)
    denominator = tf.reduce_sum(weights * E)

    # Tránh chia cho 0
    kappa = 1.0 - numerator / (denominator + 1e-8)

    return kappa


class SoftKappaLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, name="soft_kappa_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        kappa = soft_quadratic_weighted_kappa(
            y_true=y_true,
            y_pred_softmax=y_pred,
            num_classes=self.num_classes
        )
        return 1.0 - kappa  # Vì chúng ta muốn giảm loss => tối ưu hóa 1 - kappa


alpha = 0.5  # Trọng số giữa CE và Kappa Loss
cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
kappa_loss_fn = SoftKappaLoss(num_classes=5)  # Thay bằng số lớp của bạn

def combined_loss(y_true, y_pred):
    ce_loss = cce(y_true, y_pred)
    kappa_loss = kappa_loss_fn(y_true, y_pred)
    return alpha * ce_loss + (1 - alpha) * kappa_loss


@tf.function
def qwk_metric(y_true, y_pred):
    return soft_quadratic_weighted_kappa(
        y_true=y_true,
        y_pred_softmax=y_pred,
        num_classes=5  # Thay bằng số lớp thực tế
    )

loss = combined_loss(y_true, y_pred).numpy()
qwk = qwk_metric(y_true, y_pred).numpy()

model.compile(
    optimizer='adam',
    loss=combined_loss,
    metrics=['accuracy', qwk_metric]
)