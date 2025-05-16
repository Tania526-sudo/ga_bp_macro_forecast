import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

# 1. Data loading
df = pd.read_csv("macro_data.csv")  # CSV: 'IPC', 'IOC', 'KVVE', 'M0', 'M2'

# 2. Construction of logs
def create_lag_features(df, lags):
    df_lagged = pd.DataFrame()
    for col, lag in lags.items():
        df_lagged[f"{col}(lag)"] = df[col].shift(lag)
    df_lagged["IPC(+1)"] = df["IPC"].shift(-1)  
    df_lagged.dropna(inplace=True)
    return df_lagged

lags = {
    "IOC": 0,
    "IPC": 0,
    "KVVE": 7,
    "M0": 7,
    "M2": 7
}

data = create_lag_features(df, lags)
X = data.drop("IPC(+1)", axis=1).values
y = data["IPC(+1)"].values

# Scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 3. ПNeural network parameters
input_dim = X.shape[1]
mlp = MLPRegressor(hidden_layer_sizes=(5,), activation='relu', max_iter=1, warm_start=True)

# 4. Genetic algorithm
def eval_nn(individual):
    # Assign model weights from individual vector
    start = 0
    i_weights = []
    for coef in mlp.coefs_ + mlp.intercepts_:
        shape = coef.shape
        size = np.prod(shape)
        new_weights = np.array(individual[start:start + size]).reshape(shape)
        i_weights.append(new_weights)
        start += size

    mlp.coefs_ = i_weights[:len(mlp.coefs_)]
    mlp.intercepts_ = i_weights[len(mlp.coefs_):]

    y_pred = mlp.predict(X_scaled)
    return mean_squared_error(y_scaled, y_pred),

# Initialization DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

n_weights = sum(w.size for w in mlp.coefs_ + mlp.intercepts_)
toolbox.register("attr_float", lambda: random.uniform(-1.0, 1.0))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_weights)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_nn)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 5. Launching GA
pop = toolbox.population(n=30)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)

result, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=40, stats=stats, halloffame=hof, verbose=True)

# 6. Building a forecast graph
# Set the best weights
eval_nn(hof[0])
y_pred_scaled = mlp.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

plt.figure(figsize=(10, 5))
plt.plot(y[:50], label="Real IPC(+1)")
plt.plot(y_pred[:50], label="Predicted IPC(+1)")
plt.title("Forecasting the Consumer Price Index (IPC+1)")
plt.xlabel("Time")
plt.ylabel("Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. MSE
mse = mean_squared_error(y, y_pred)
print(f"Final MSE: {mse:.4f}")

