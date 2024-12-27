from MLP import MLP


# Dataset XOR
X_train = [
    [0.0, 0.0],  # False False -> False
    [0.0, 1.0],  # False True  -> True
    [1.0, 0.0],  # True False  -> True
    [1.0, 1.0]   # True True   -> False
]

y_train = [
    [0.0],  # 0 XOR 0 = 0
    [1.0],  # 0 XOR 1 = 1
    [1.0],  # 1 XOR 0 = 1
    [0.0]   # 1 XOR 1 = 0
]

mlp = MLP(input_size=2, layers_size=[4, 4, 1])

for epoch in range(1000):
    total_loss = 0.0

    for x, y in zip(X_train, y_train):
        loss = mlp.train_step(x, y)
        total_loss += loss

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(X_train)}")


print(f"\n Prediction test:")
for x in X_train:
    pred = mlp(x)[0].data
    print(f"Input: {x}, Prediction: {pred:.3f}")