import networkx as nx
import numpy as np
import tensorflow as tf
from qiskit_aer import AerSimulator
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import Sampler  # Switch to base Sampler
from flask import Flask, request, render_template, jsonify

# Step 1: Create a simple payment network (undirected graph)
G = nx.Graph()
banks = ["A", "B", "C", "D", "E"]
G.add_nodes_from(banks)
edges = [("A", "B", {"latency": 10, "fee": 2, "liquidity": 10000}),
         ("A", "C", {"latency": 15, "fee": 3, "liquidity": 8000}),
         ("B", "D", {"latency": 20, "fee": 1, "liquidity": 12000}),
         ("C", "D", {"latency": 12, "fee": 4, "liquidity": 6000}),
         ("D", "E", {"latency": 18, "fee": 2, "liquidity": 9000}),
         ("B", "E", {"latency": 25, "fee": 3, "liquidity": 7000})]
G.add_edges_from(edges)

# Step 2: Simple AI model to predict a starting path
def train_ai_model():
    X = np.array([[0, 4, 100, 2, 10000], [0, 4, 150, 3, 8000], [1, 4, 250, 3, 7000]])
    y = np.array([0.5, 0.7, 0.9])  # Lower score = better
    model = tf.keras.Sequential([tf.keras.layers.Dense(5, activation='relu', input_shape=(5,)),
                                 tf.keras.layers.Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)
    return model

ai_model = train_ai_model()

def ai_predict(start, end):
    paths = list(nx.all_simple_paths(G, start, end, cutoff=3))  # Limit to 3 hops
    if not paths:
        return None
    inputs = []
    for path in paths:
        total_latency, total_fee, min_liquidity = 0, 0, float('inf')
        for i in range(len(path) - 1):
            edge = G[path[i]][path[i+1]]
            total_latency += edge["latency"]
            total_fee += edge["fee"]
            min_liquidity = min(min_liquidity, edge["liquidity"])
        inputs.append([banks.index(start), banks.index(end), total_latency, total_fee, min_liquidity])
    scores = ai_model.predict(np.array(inputs), verbose=0)
    best_idx = np.argmin(scores)
    return paths[best_idx]

# Step 3: Quantum optimization with fixed QAOA
def quantum_optimize(start, end, ai_path):
    try:
        paths = list(nx.all_simple_paths(G, start, end, cutoff=3))
        if not paths or len(paths) < 2:
            print(f"Quantum skipped: Only {len(paths)} path(s) found")
            return ai_path

        print(f"Quantum running: Found {len(paths)} paths to optimize")
        for i, path in enumerate(paths):
            print(f"Path {i}: {' -> '.join(path)}")

        # Define a simple cost function
        qp = QuadraticProgram("payment_routing")
        for i in range(len(paths)):
            qp.binary_var(f"x_{i}")
        
        # Calculate costs for each path
        linear_costs = {}
        for i, path in enumerate(paths):
            total_cost = 0
            for j in range(len(path) - 1):
                edge_data = G[path[j]][path[j+1]]
                total_cost += (edge_data["latency"] / 1000) + (edge_data["fee"] / 10) - (edge_data["liquidity"] / 100000)
            linear_costs[f"x_{i}"] = total_cost
            print(f"Cost for Path {i}: {total_cost}")
        
        qp.minimize(linear=linear_costs)
        qp.linear_constraint(linear={f"x_{i}": 1 for i in range(len(paths))}, sense="==", rhs=1)

        # Run QAOA with updated API
        backend = AerSimulator(method='statevector')
        sampler = Sampler()  # Base Sampler, no backend in constructor
        optimizer = SPSA(maxiter=50)
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
        qaoa_meo = MinimumEigenOptimizer(qaoa)
        result = qaoa_meo.solve(qp)
        
        best_idx = [i for i, val in enumerate(result.x) if val == 1][0]
        print(f"Quantum chose Path {best_idx}: {' -> '.join(paths[best_idx])}")
        return paths[best_idx]
    except Exception as e:
        print(f"Quantum error: {str(e)}")
        return ai_path  # Fallback to AI path if quantum fails

# Flask web interface
app = Flask(__name__)
@app.route("/", methods=["GET"])
def home():
    # For the initial page load, show the nodes for selection
    nodes = banks  # Use your existing banks list
    return render_template("index.html", nodes=nodes)

@app.route("/route", methods=["POST"])
def route_payment():
    try:
        data = request.form
        start = data["start"].upper()
        end = data["end"].upper()
        amount = float(data["amount"])
        
        if start not in banks or end not in banks:
            return render_template("result.html", 
                                  path="Error: Invalid bank name!", 
                                  amount=amount,
                                  total_cost=0)
        
        ai_path = ai_predict(start, end)
        if not ai_path:
            return render_template("result.html", 
                                  path="Error: No path found between banks!", 
                                  amount=amount,
                                  total_cost=0)
        
        quantum_path = quantum_optimize(start, end, ai_path)
        
        # Calculate the cost for the quantum path
        quantum_cost = sum(G[quantum_path[i]][quantum_path[i+1]]["latency"] + G[quantum_path[i]][quantum_path[i+1]]["fee"]
                           for i in range(len(quantum_path) - 1))
        
        # Render the result template with the final path
        path_display = " -> ".join(quantum_path)
        
        return render_template("result.html", 
                              amount=amount, 
                              path=path_display, 
                              total_cost=quantum_cost)
    except Exception as e:
        return render_template("result.html", 
                              path=f"Error: {str(e)}", 
                              amount=amount if 'amount' in locals() else 0,
                              total_cost=0)
'''
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/route", methods=["POST"])
def route_payment():
    try:
        data = request.form
        start = data["start"].upper()
        end = data["end"].upper()
        amount = float(data["amount"])
        
        if start not in banks or end not in banks:
            return jsonify({"error": "Invalid bank name!"})
        
        ai_path = ai_predict(start, end)
        if not ai_path:
            return jsonify({"error": "No path found between banks!"})
        
        quantum_path = quantum_optimize(start, end, ai_path)
        
        ai_cost = sum(G[ai_path[i]][ai_path[i+1]]["latency"] + G[ai_path[i]][ai_path[i+1]]["fee"]
                      for i in range(len(ai_path) - 1))
        quantum_cost = sum(G[quantum_path[i]][quantum_path[i+1]]["latency"] + G[quantum_path[i]][quantum_path[i+1]]["fee"]
                           for i in range(len(quantum_path) - 1))
        
        result = {
            "ai_path": " -> ".join(ai_path),
            "ai_cost": ai_cost,
            "quantum_path": " -> ".join(quantum_path),
            "quantum_cost": quantum_cost
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})'''

if __name__ == "__main__":
    app.run(debug=True)

'''from flask import Flask, request, render_template
import numpy as np
import networkx as nx
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import StatevectorSampler
from qiskit import QuantumCircuit
import tensorflow as tf

app = Flask(__name__)

def create_payment_network():
    G = nx.Graph()
    G.add_nodes_from(["NY Bank", "London Hub", "Tokyo Processor", "Sydney Node", 
                     "Singapore Exchange", "Frankfurt Center", "Mumbai Relay"])
    
    G.add_edge("NY Bank", "London Hub", cost=5.0, latency=1.5)
    G.add_edge("NY Bank", "Tokyo Processor", cost=8.0, latency=3.0)
    G.add_edge("NY Bank", "Frankfurt Center", cost=6.0, latency=1.8)
    G.add_edge("London Hub", "Tokyo Processor", cost=4.0, latency=2.0)
    G.add_edge("London Hub", "Sydney Node", cost=7.0, latency=2.5)
    G.add_edge("London Hub", "Frankfurt Center", cost=2.0, latency=0.8)
    G.add_edge("London Hub", "Mumbai Relay", cost=5.5, latency=2.2)
    G.add_edge("Tokyo Processor", "Sydney Node", cost=3.0, latency=1.0)
    G.add_edge("Tokyo Processor", "Singapore Exchange", cost=2.5, latency=1.2)
    G.add_edge("Sydney Node", "Singapore Exchange", cost=2.8, latency=1.1)
    G.add_edge("Singapore Exchange", "Mumbai Relay", cost=3.2, latency=1.3)
    G.add_edge("Frankfurt Center", "Mumbai Relay", cost=4.8, latency=1.9)
    
    return G

def ai_predict_weights(G):
    costs = []
    latencies = []
    for u, v, data in G.edges(data=True):
        costs.append(data["cost"])
        latencies.append(data["latency"])
    
    mean_cost = np.mean(costs)
    std_cost = np.std(costs)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    num_synthetic = 50
    np.random.seed(42)
    synthetic_costs = np.maximum(0.5, np.random.normal(mean_cost, std_cost, num_synthetic))
    synthetic_latencies = np.maximum(0.3, np.random.normal(mean_latency, std_latency, num_synthetic))
    
    base_cost = 1.0
    cost_factor = 0.8
    latency_factor = 1.5
    synthetic_weights = base_cost + (cost_factor * synthetic_costs) + (latency_factor * synthetic_latencies)
    
    X = np.column_stack((synthetic_costs, synthetic_latencies))
    y = synthetic_weights
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="relu")
    ])
    
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=50, verbose=0, validation_split=0.2)
    
    weights = {}
    for u, v, data in G.edges(data=True):
        pred = model.predict(np.array([[data["cost"], data["latency"]]]), verbose=0)[0][0]
        pred = max(0.5, pred)
        weights[(u, v)] = pred
        weights[(v, u)] = pred
    
    return weights

def apply_ising_hamiltonian(circuit, operator, gamma, num_qubits):
    """Apply the Ising Hamiltonian as circuit gates."""
    for i in range(len(operator)):
        pauli_str = operator.paulis[i].to_label()
        coeff = operator.coeffs[i].real
        for qubit, pauli_op in enumerate(pauli_str[::-1]):
            if pauli_op == 'Z':
                circuit.rz(2 * gamma * coeff, qubit)
            # Handle ZZ terms if present
            elif pauli_op == 'Z' and qubit < num_qubits - 1:
                next_op = pauli_str[::-1][qubit + 1]
                if next_op == 'Z':
                    circuit.cx(qubit, qubit + 1)
                    circuit.rz(2 * gamma * coeff, qubit + 1)
                    circuit.cx(qubit, qubit + 1)

def quantum_route_optimization(G, weights, start, end):
    print(f"Finding route from {start} to {end}")
    
    G_weighted = G.copy()
    for u, v in G.edges():
        G_weighted[u][v]['weight'] = weights.get((u, v), 10.0)
    
    try:
        fallback_path = nx.shortest_path(G_weighted, source=start, target=end, weight='weight')
        fallback_cost = sum(weights.get((fallback_path[i], fallback_path[i+1]), 0) for i in range(len(fallback_path)-1))
        print(f"Fallback path: {fallback_path}, cost: {fallback_cost}")
    except nx.NetworkXNoPath:
        fallback_path = [start, end]
        fallback_cost = weights.get((start, end), 10.0)
    
    try:
        G_directed = G.to_directed()
        
        qp = QuadraticProgram()
        edge_vars = {}
        edges = list(G.edges())
        num_vars = len(edges)
        for idx, (u, v) in enumerate(edges):
            var_name = f"x_{idx}"
            qp.binary_var(var_name)
            edge_vars[(u, v)] = var_name
        
        obj_linear = {edge_vars[(u, v)]: weights.get((u, v), 0) for (u, v) in edge_vars}
        qp.minimize(linear=obj_linear)
        
        nodes = list(G.nodes())
        for node in nodes:
            if node != start and node != end:
                flow_constraint = {}
                for u, v in G_directed.edges():
                    if v == node and (u, v) in edge_vars:
                        flow_constraint[edge_vars[(u, v)]] = 1
                    elif u == node and (u, v) in edge_vars:
                        flow_constraint[edge_vars[(u, v)]] = -1
                if flow_constraint:
                    qp.linear_constraint(flow_constraint, "==", 0)
        
        start_constraint = {edge_vars[(start, v)]: 1 for start, v in G_directed.edges(start) if (start, v) in edge_vars}
        if start_constraint:
            qp.linear_constraint(start_constraint, "==", 1)
        
        end_constraint = {edge_vars[(u, end)]: 1 for u, end in G_directed.edges(end) if (u, end) in edge_vars}
        if end_constraint:
            qp.linear_constraint(end_constraint, "==", 1)
        
        # Convert to QUBO and Ising
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        operator, offset = qubo.to_ising()
        
        # Manual QAOA implementation
        num_qubits = operator.num_qubits
        circuit = QuantumCircuit(num_qubits)
        circuit.h(range(num_qubits))  # Initial superposition
        
        # QAOA parameters (p=1)
        gamma = 0.0
        beta = 0.0
        optimizer = SPSA(maxiter=20)
        
        def objective_function(params):
            nonlocal gamma, beta
            gamma, beta = params[0], params[1]
            qc = QuantumCircuit(num_qubits)
            qc.h(range(num_qubits))
            apply_ising_hamiltonian(qc, operator, gamma, num_qubits)
            qc.rx(2 * beta, range(num_qubits))  # Mixer
            
            sampler = StatevectorSampler()
            job = sampler.run([qc])  # Removed parameter_values
            result = job.result()
            quasi_dist = result[0].data.evs
            if isinstance(quasi_dist, (float, np.ndarray)):
                return quasi_dist + offset
            else:
                raise ValueError("Invalid quasi-distribution format")
        
        # Optimize parameters
        initial_params = np.array([1.0, 1.0])
        opt_result = optimizer.minimize(objective_function, initial_params)
        optimal_params = opt_result.x
        
        # Final circuit with optimized parameters
        final_circuit = QuantumCircuit(num_qubits)
        final_circuit.h(range(num_qubits))
        apply_ising_hamiltonian(final_circuit, operator, optimal_params[0], num_qubits)
        final_circuit.rx(2 * optimal_params[1], range(num_qubits))
        final_circuit.measure_all()
        
        # Sample the final circuit
        sampler = StatevectorSampler()
        job = sampler.run([final_circuit])  # Removed parameter_values
        result = job.result()
        counts = result[0].data.meas.get_counts()
        
        # Get the most probable bitstring
        bitstring = max(counts, key=counts.get)
        bitstring = [int(bit) for bit in bitstring]
        
        selected_edges = [(u, v) for (u, v), var_name in edge_vars.items() 
                         if bitstring[list(qp.variables).index(qp.get_variable(var_name))] == 1]
        
        if not selected_edges:
            print("No edges selected in quantum solution")
            return fallback_path, fallback_cost
        
        path = [start]
        current = start
        visited = set([start])
        max_iterations = len(nodes) * 2
        
        for _ in range(max_iterations):
            for u, v in selected_edges:
                if u == current and v not in visited:
                    path.append(v)
                    visited.add(v)
                    current = v
                    break
            else:
                try:
                    completion_path = nx.shortest_path(G_weighted, source=current, target=end, weight='weight')
                    path.extend(completion_path[1:])
                    break
                except:
                    print("Could not complete partial quantum path")
                    return fallback_path, fallback_cost
            if current == end:
                break
        
        if current != end:
            print("Path construction did not reach the end node")
            return fallback_path, fallback_cost
        
        total_cost = sum(weights.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
        print(f"Quantum path: {path}, cost: {total_cost}")
        
        if len(path) >= 2 and path[0] == start and path[-1] == end and total_cost > 0:
            return path, total_cost
        return fallback_path, fallback_cost
        
    except Exception as e:
        print(f"Error in quantum optimization: {e}")
        return fallback_path, fallback_cost

@app.route("/", methods=["GET", "POST"])
def home():
    G = create_payment_network()
    nodes = list(G.nodes)
    if request.method == "POST":
        start = request.form["start"]
        end = request.form["end"]
        amount = float(request.form["amount"])

        if start == end:
            return render_template("result.html", 
                                  amount=amount, 
                                  path="Error: Start and end nodes must be different", 
                                  total_cost=0.0)

        weights = ai_predict_weights(G)
        path, total_cost = quantum_route_optimization(G, weights, start, end)
        
        path_display = " -> ".join(path) if path else "No valid path found"
        
        return render_template("result.html", 
                              amount=amount, 
                              path=path_display, 
                              total_cost=total_cost)
    return render_template("index.html", nodes=nodes)

if __name__ == "__main__":
    app.run(debug=True)
    '''