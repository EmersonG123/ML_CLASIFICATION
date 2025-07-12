import matplotlib.pyplot as plt

# Épocas del 1 al 16
epochs = list(range(1, 17))

# Datos actualizados
train_loss = [1.6866, 1.1702, 1.0126, 0.9125, 0.8500, 0.7727, 0.7297, 0.6735, 0.6546, 0.6269, 0.6071, 0.5992, 0.5396, 0.5045, 0.5191]
val_loss   = [1.2439, 1.1150, 1.0699, 1.0017, 0.9754, 0.9454, 1.1304, 0.9815, 0.9548, 1.1080, 1.0116, 1.0125, 1.0704, 0.9515, 0.9520]

train_acc  = [0.4104, 0.5955, 0.6506, 0.6764, 0.6978, 0.7176, 0.7335, 0.7509, 0.7583, 0.7777, 0.7772, 0.7643, 0.7980, 0.8228, 0.8010]
val_acc    = [0.6071, 0.6562, 0.6295, 0.6429, 0.6429, 0.7098, 0.6116, 0.6920, 0.6741, 0.6473, 0.6696, 0.6518, 0.6875, 0.6741, 0.6696]

# Crear las gráficas
plt.figure(figsize=(12, 5))

# Pérdida
plt.subplot(1, 2, 1)
plt.plot(epochs[:15], train_loss, label='Train Loss', marker='o', color='blue')
plt.plot(epochs[:15], val_loss, label='Val Loss', marker='o', color='orange')
plt.title('Evolución de la Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Precisión
plt.subplot(1, 2, 2)
plt.plot(epochs[:15], train_acc, label='Train Accuracy', marker='o', color='green')
plt.plot(epochs[:15], val_acc, label='Val Accuracy', marker='o', color='red')
plt.title('Evolución de la Precisión')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# Mostrar ambas gráficas
plt.tight_layout()
plt.show()
