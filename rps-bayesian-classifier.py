import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io
from scipy.ndimage import binary_fill_holes

folder = Path("Rock Paper Scissors")
all_pictures = sorted(folder.rglob("*.png"))


def preprocessing(path):
    img = io.imread(path)
    R = img[:, :, 0].astype(float)
    G = img[:, :, 1].astype(float)

    mask_bin = binary_fill_holes(R - G > 15)
    rows, cols = np.where(mask_bin == 1)
    mask_crop = mask_bin[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    return (~mask_crop).astype(int)


feats = ["hand_ratio", "centre", "left", "right", "left_third", "aspect_ratio"]


def feature_extraction(mask_crop):
    hand = (mask_crop == 0)
    H, W = hand.shape
    hand_ratio = hand.mean()
    centre = hand[H // 4:3 * H // 4, W // 4:3 * W // 4].mean()
    m_W = max(1, int(0.1 * W))
    left = hand[:, :m_W].mean()
    right = hand[:, -m_W:].mean()
    left_third = hand[:, :W // 3].mean()
    aspect_ratio = W / H
    return [hand_ratio, centre, left, right, left_third, aspect_ratio]


F = []
y = []

for pic in all_pictures:
    img_processed = preprocessing(pic)
    feat = feature_extraction(img_processed)
    F.append(feat)
    y.append(pic.parent.name)

F = np.array(F)
y = np.array(y)

P = F[y == "paper"]
S = F[y == "scissors"]
R = F[y == "rock"]


def divide(X):
    np.random.seed(42)
    idX = np.random.permutation(len(X))
    limit = int(len(X) * 0.8)
    return X[idX[:limit]], X[idX[limit:]]


P_train, P_test = divide(P)
S_train, S_test = divide(S)
R_train, R_test = divide(R)

X_train = np.concatenate([P_train, S_train, R_train])
y_train = np.array([0] * len(P_train) + [1] * len(S_train) + [2] * len(R_train))

X_test = np.concatenate([P_test, S_test, R_test])
y_test = np.array([0] * len(P_test) + [1] * len(S_test) + [2] * len(R_test))

mu_P = np.mean(P_train, axis=0)
mu_S = np.mean(S_train, axis=0)
mu_R = np.mean(R_train, axis=0)

reg = 1e-5 * np.eye(6)
Sigma_P = np.cov(P_train.T) + reg
Sigma_S = np.cov(S_train.T) + reg
Sigma_R = np.cov(R_train.T) + reg

inv_P = np.linalg.inv(Sigma_P)
inv_S = np.linalg.inv(Sigma_S)
inv_R = np.linalg.inv(Sigma_R)

logdet_P = np.log(np.linalg.det(Sigma_P))
logdet_S = np.log(np.linalg.det(Sigma_S))
logdet_R = np.log(np.linalg.det(Sigma_R))

total_tr = len(P_train) + len(S_train) + len(R_train)

prior_P = np.log(len(P_train) / total_tr)
prior_S = np.log(len(S_train) / total_tr)
prior_R = np.log(len(R_train) / total_tr)


def discriminant_function(x, mu, inv_Sigma, logdet, log_prior):
    diff = x - mu
    return -0.5 * logdet - 0.5 * (diff @ inv_Sigma @ diff) + log_prior


def predict(X):
    preds = []
    for x in X:
        g0 = discriminant_function(x, mu_P, inv_P, logdet_P, prior_P)
        g1 = discriminant_function(x, mu_S, inv_S, logdet_S, prior_S)
        g2 = discriminant_function(x, mu_R, inv_R, logdet_R, prior_R)
        preds.append(np.argmax([g0, g1, g2]))
    return np.array(preds)


y_train_pred = predict(X_train)
y_test_pred = predict(X_test)


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm


cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print(f"Train accuracy: {np.mean(y_train == y_train_pred) * 100:.2f}%")
print(f"Test accuracy:  {np.mean(y_test == y_test_pred) * 100:.2f}%")

names = ["paper", "scissors", "rock"]

# ---------------- 1. Preprocessed Masks ----------------
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for ax, pic in zip(axes, all_pictures[::400][:5]):
    mask = preprocessing(pic)
    ax.imshow(mask, cmap="gray")
    ax.set_title(pic.parent.name)
    ax.axis("off")

plt.tight_layout()
plt.savefig("preprocessing.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------- 2. 2D Decision Boundaries (Classifier) ----------------
i, j = 0, 5  # hand_ratio vs aspect_ratio
P2, S2, R2 = P_train[:, [i, j]], S_train[:, [i, j]], R_train[:, [i, j]]
X_te2 = X_test[:, [i, j]]

mu_P2, mu_S2, mu_R2 = np.mean(P2, axis=0), np.mean(S2, axis=0), np.mean(R2, axis=0)
reg2 = 1e-5 * np.eye(2)

inv_P2, ld_P2 = np.linalg.inv(np.cov(P2.T) + reg2), np.log(np.linalg.det(np.cov(P2.T) + reg2))
inv_S2, ld_S2 = np.linalg.inv(np.cov(S2.T) + reg2), np.log(np.linalg.det(np.cov(S2.T) + reg2))
inv_R2, ld_R2 = np.linalg.inv(np.cov(R2.T) + reg2), np.log(np.linalg.det(np.cov(R2.T) + reg2))

xx, yy = np.meshgrid(
    np.linspace(X_te2[:, 0].min() - 0.05, X_te2[:, 0].max() + 0.05, 250),
    np.linspace(X_te2[:, 1].min() - 0.05, X_te2[:, 1].max() + 0.05, 250),
)

Z = []
for pt in np.c_[xx.ravel(), yy.ravel()]:
    g0 = discriminant_function(pt, mu_P2, inv_P2, ld_P2, prior_P)
    g1 = discriminant_function(pt, mu_S2, inv_S2, ld_S2, prior_S)
    g2 = discriminant_function(pt, mu_R2, inv_R2, ld_R2, prior_R)
    Z.append(np.argmax([g0, g1, g2]))
Z = np.array(Z).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#a8ddb5", "#ccebc5", "#fde0dd"], alpha=0.7)
plt.contour(xx, yy, Z, levels=[0.5, 1.5], colors="#2b5c3f", linewidths=1.5)

colors = ["#1b7837", "#386cb0", "#d95f02"]
for k in range(3):
    plt.scatter(X_te2[y_test == k, 0], X_te2[y_test == k, 1], c=colors[k], label=names[k], edgecolors="white", s=45)

plt.scatter(
    [mu_P2[0], mu_S2[0], mu_R2[0]],
    [mu_P2[1], mu_S2[1], mu_R2[1]],
    c="gold", marker="*", s=260, edgecolors="black", linewidths=1.2, label="Class means (μ)",
)

plt.xlabel(feats[i], fontsize=11)
plt.ylabel(feats[j], fontsize=11)
plt.title(f"Decision Boundaries ({feats[i]} vs {feats[j]})", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig("decision_boundaries.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------- 3. Confusion Matrices ----------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, cm, title in zip(axes, [cm_train, cm_test], ["Train data", "Test data"]):
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(3):
        for j in range(3):
            col = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=col, fontweight="bold")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------- 4. Misclassified Examples (True vs Pred) ----------------
misclassified = []
for pic in all_pictures:
    true_cls = pic.parent.name
    mask = preprocessing(pic)
    f = feature_extraction(mask)
    pred_cls = names[predict([f])[0]]
    if true_cls != pred_cls:
        misclassified.append((pic, true_cls, pred_cls, mask))
    if len(misclassified) == 4:
        break

if misclassified:
    fig, axes = plt.subplots(2, len(misclassified), figsize=(10, 5))
    for k, (pic, true_cls, pred_cls, mask) in enumerate(misclassified):
        axes[0, k].imshow(io.imread(pic))
        axes[0, k].set_title(f"True: {true_cls}\nPred: {pred_cls}", color="crimson", fontweight="bold")
        axes[0, k].axis("off")

        axes[1, k].imshow(mask, cmap="gray")
        axes[1, k].axis("off")
    plt.tight_layout()
    plt.savefig("misclassified.png", dpi=200, bbox_inches="tight")
    plt.show()