'''# payment_router.py
from flask import Flask, request, render_template
import numpy as np
import networkx as nx
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA
from qiskit_aer import AerSimulator
import tensorflow as tf
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit, transpile

app = Flask(__name__)

def create_payment_network():
    G = nx.Graph()
    G.add_nodes_from(["NY Bank", "London Hub", "Tokyo Processor", "Sydney Node"])
    G.add_edge("NY Bank", "London Hub", cost=5.0, latency=1.5)
    G.add_edge("NY Bank", "Tokyo Processor", cost=8.0, latency=3.0)
    G.add_edge("London Hub", "Tokyo Processor", cost=4.0, latency=2.0)
    G.add_edge("London Hub", "Sydney Node", cost=7.0, latency=2.5)
    G.add_edge("Tokyo Processor", "Sydney Node", cost=3.0, latency=1.0)
    return G

def ai_predict_weights(G):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="relu", input_shape=(2,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    X = np.array([[5.0, 1.5], [8.0, 3.0], [4.0, 2.0], [7.0, 2.5], [3.0, 1.0]])
    y = np.array([4.0, 6.0, 3.5, 5.5, 2.5])
    model.fit(X, y, epochs=10, verbose=0)
    weights = {}
    for u, v, data in G.edges(data=True):
        pred = model.predict(np.array([[data["cost"], data["latency"]]]), verbose=0)[0][0]
        weights[(u, v)] = pred
        weights[(v, u)] = pred
    return weights

def quantum_route_optimization(G, weights, start, end):
    qp = QuadraticProgram()
    nodes = list(G.nodes)
    for i, node in enumerate(nodes):
        qp.binary_var(f"x_{i}")
    
    linear = {}
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if (u, v) in weights:
                linear[f"x_{i}"] = weights[(u, v)]
    qp.minimize(linear=linear)
    
    start_idx = nodes.index(start)
    end_idx = nodes.index(end)
    qp.linear_constraint({f"x_{start_idx}": 1}, "==", 1)
    qp.linear_constraint({f"x_{end_idx}": 1}, "==", 1)
    
    # Convert to QUBO and solve with QAOA
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)
    operator, offset = qubo.to_ising()
    
    # Set up QAOA with Sampler
    optimizer = SPSA(maxiter=50)
    sampler = Sampler()
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
    result = qaoa.compute_minimum_eigenvalue(operator)
    
    # Build and simulate the QAOA circuit with AerSimulator
    num_qubits = operator.num_qubits
    circuit = QuantumCircuit(num_qubits, num_qubits)  # Add classical bits
    circuit.h(range(num_qubits))  # Initial superposition
    circuit.append(qaoa.ansatz.bind_parameters(result.optimal_parameters), range(num_qubits), range(num_qubits))
    
    # Simulate with AerSimulator
    simulator = AerSimulator()
    circuit.measure_all()  # This will measure to the classical bits
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result_sim = job.result()
    counts = result_sim.get_counts()
    
    # Get the most frequent bitstring and clean it
    best_sample = max(counts.items(), key=lambda item: item[1])[0]  # Bitstring with highest count
    clean_sample = best_sample.replace(" ", "")  # Remove spaces
    # Limit to the first len(nodes) qubits (original variables)
    clean_sample = clean_sample[:len(nodes)]
    path_indices = [i for i, val in enumerate(clean_sample) if int(val) == 1]
    path = [nodes[i] for i in path_indices]
    total_cost = sum(weights.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
    return path, total_cost

@app.route("/", methods=["GET", "POST"])
def home():
    G = create_payment_network()
    nodes = list(G.nodes)
    if request.method == "POST":
        start = request.form["start"]
        end = request.form["end"]
        amount = float(request.form["amount"])

        weights = ai_predict_weights(G)
        path, total_cost = quantum_route_optimization(G, weights, start, end)
        
        return render_template("result.html", 
                              amount=amount, 
                              path=" -> ".join(path), 
                              total_cost=total_cost)
    return render_template("index.html", nodes=nodes)

if __name__ == "__main__":
    app.run(debug=True)'
'''
'''# payment_router.py
from flask import Flask, request, render_template
import numpy as np
import networkx as nx
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA
from qiskit_aer import AerSimulator
import tensorflow as tf
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit, transpile

app = Flask(__name__)

def create_payment_network():
    G = nx.Graph()
    G.add_nodes_from(["NY Bank", "London Hub", "Tokyo Processor", "Sydney Node"])
    G.add_edge("NY Bank", "London Hub", cost=5.0, latency=1.5)
    G.add_edge("NY Bank", "Tokyo Processor", cost=8.0, latency=3.0)
    G.add_edge("London Hub", "Tokyo Processor", cost=4.0, latency=2.0)
    G.add_edge("London Hub", "Sydney Node", cost=7.0, latency=2.5)
    G.add_edge("Tokyo Processor", "Sydney Node", cost=3.0, latency=1.0)
    return G

def ai_predict_weights(G):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="relu", input_shape=(2,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    X = np.array([[5.0, 1.5], [8.0, 3.0], [4.0, 2.0], [7.0, 2.5], [3.0, 1.0]])
    y = np.array([4.0, 6.0, 3.5, 5.5, 2.5])
    model.fit(X, y, epochs=10, verbose=0)
    weights = {}
    for u, v, data in G.edges(data=True):
        pred = model.predict(np.array([[data["cost"], data["latency"]]]), verbose=0)[0][0]
        weights[(u, v)] = pred
        weights[(v, u)] = pred
    return weights

def quantum_route_optimization(G, weights, start, end):
    # For debugging
    print(f"Finding route from {start} to {end}")
    
    # As a fallback, find shortest path using Dijkstra's algorithm
    # This ensures we always return a valid path between start and end
    try:
        fallback_path = nx.shortest_path(G, source=start, target=end, weight='cost')
        fallback_cost = sum(weights.get((fallback_path[i], fallback_path[i+1]), 0) for i in range(len(fallback_path)-1))
        print(f"Fallback path: {fallback_path}, cost: {fallback_cost}")
    except nx.NetworkXNoPath:
        fallback_path = [start, end]  # Direct connection if no path found
        fallback_cost = weights.get((start, end), 10.0)  # Default high cost
    
    try:
        # Try quantum optimization approach
        qp = QuadraticProgram()
        nodes = list(G.nodes)
        
        # Create binary variables for each edge in the graph
        edge_vars = {}
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if G.has_edge(u, v):
                    var_name = f"x_{i}_{j}"
                    qp.binary_var(var_name)
                    edge_vars[(u, v)] = var_name
        
        # Objective function: minimize the total weight
        obj_linear = {}
        for (u, v), var_name in edge_vars.items():
            obj_linear[var_name] = weights.get((u, v), 0)
        qp.minimize(linear=obj_linear)
        
        # Constraints: flow conservation
        # For each node except start and end: incoming flow = outgoing flow
        for node in nodes:
            if node != start and node != end:
                flow_constraint = {}
                # Incoming edges
                for u in G.predecessors(node):
                    flow_constraint[edge_vars.get((u, node), "dummy")] = 1
                # Outgoing edges
                for v in G.successors(node):
                    flow_constraint[edge_vars.get((node, v), "dummy")] = -1
                
                qp.linear_constraint(flow_constraint, "==", 0)
        
        # Start node must have one outgoing edge
        start_constraint = {}
        for v in G.successors(start):
            start_constraint[edge_vars.get((start, v), "dummy")] = 1
        qp.linear_constraint(start_constraint, "==", 1)
        
        # End node must have one incoming edge
        end_constraint = {}
        for u in G.predecessors(end):
            end_constraint[edge_vars.get((u, end), "dummy")] = 1
        qp.linear_constraint(end_constraint, "==", 1)
        
        # Convert to QUBO and solve with QAOA
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        operator, offset = qubo.to_ising()
        
        # Set up QAOA with Sampler
        optimizer = SPSA(maxiter=50)
        sampler = Sampler()
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
        result = qaoa.compute_minimum_eigenvalue(operator)
        
        # Get the most probable bitstring
        bitstring = max(result.eigenstate, key=result.eigenstate.get)
        
        # Reconstruct the path
        selected_edges = []
        for (u, v), var_name in edge_vars.items():
            var_idx = list(qp.variables).index(qp.get_variable(var_name))
            if bitstring[var_idx] == 1:
                selected_edges.append((u, v))
        
        # Construct the path from selected edges
        path = [start]
        current = start
        visited = set([start])
        
        while current != end:
            for u, v in selected_edges:
                if u == current and v not in visited:
                    path.append(v)
                    visited.add(v)
                    current = v
                    break
            else:
                print("Could not construct complete path from quantum solution")
                return fallback_path, fallback_cost
        
        total_cost = sum(weights.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
        print(f"Quantum path: {path}, cost: {total_cost}")
        return path, total_cost
        
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

        # Don't allow same start and end
        if start == end:
            return render_template("result.html", 
                                  amount=amount, 
                                  path="Error: Start and end nodes must be different", 
                                  total_cost=0.0)

        weights = ai_predict_weights(G)
        path, total_cost = quantum_route_optimization(G, weights, start, end)
        
        return render_template("result.html", 
                              amount=amount, 
                              path=" -> ".join(path), 
                              total_cost=total_cost)
    return render_template("index.html", nodes=nodes)

if __name__ == "__main__":
    app.run(debug=True)'''
# payment_router.py
from flask import Flask, request, render_template
import numpy as np
import networkx as nx
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA
from qiskit_aer import AerSimulator
import tensorflow as tf
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit, transpile

app = Flask(__name__)

def create_payment_network():
    G = nx.Graph()
    G.add_nodes_from(["NY Bank", "London Hub", "Tokyo Processor", "Sydney Node", 
                     "Singapore Exchange", "Frankfurt Center", "Mumbai Relay"])
    
    # Add edges with cost and latency attributes
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
    # Create a larger synthetic dataset based on network properties
    # Collect real edge data from graph
    costs = []
    latencies = []
    for u, v, data in G.edges(data=True):
        costs.append(data["cost"])
        latencies.append(data["latency"])
    
    # Calculate statistics to generate more realistic data
    mean_cost = np.mean(costs)
    std_cost = np.std(costs)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    # Generate synthetic data with similar properties to our real edges
    num_synthetic = 50
    np.random.seed(42)  # For reproducibility
    synthetic_costs = np.random.normal(mean_cost, std_cost, num_synthetic)
    synthetic_latencies = np.random.normal(mean_latency, std_latency, num_synthetic)
    
    # Ensure positive values
    synthetic_costs = np.maximum(0.5, synthetic_costs)
    synthetic_latencies = np.maximum(0.3, synthetic_latencies)
    
    # Create target weights based on a reasonable formula (lower cost and latency = better)
    # Weight = base_cost + (cost_factor * cost) + (latency_factor * latency)
    base_cost = 1.0
    cost_factor = 0.8
    latency_factor = 1.5
    synthetic_weights = base_cost + (cost_factor * synthetic_costs) + (latency_factor * synthetic_latencies)
    
    # Create training data
    X = np.column_stack((synthetic_costs, synthetic_latencies))
    y = synthetic_weights
    
    # Build and train model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(2,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="relu")  # ReLU ensures positive outputs
    ])
    
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=50, verbose=0, validation_split=0.2)
    
    # Predict weights for actual graph edges
    weights = {}
    for u, v, data in G.edges(data=True):
        # Predict and ensure positive weight with minimum value
        pred = model.predict(np.array([[data["cost"], data["latency"]]]), verbose=0)[0][0]
        pred = max(0.5, pred)  # Ensure minimum positive weight
        weights[(u, v)] = pred
        weights[(v, u)] = pred  # Graph is undirected
    
    return weights

def quantum_route_optimization(G, weights, start, end):
    # For debugging
    print(f"Finding route from {start} to {end}")
    
    # Calculate shortest path using Dijkstra's algorithm as a fallback
    try:
        # Create a copy of the graph with the predicted weights
        G_weighted = G.copy()
        for u, v in G.edges():
            G_weighted[u][v]['weight'] = weights.get((u, v), 10.0)
        
        fallback_path = nx.shortest_path(G_weighted, source=start, target=end, weight='weight')
        fallback_cost = sum(weights.get((fallback_path[i], fallback_path[i+1]), 0) for i in range(len(fallback_path)-1))
        print(f"Fallback path: {fallback_path}, cost: {fallback_cost}")
    except nx.NetworkXNoPath:
        fallback_path = [start, end]  # Direct connection if no path found
        fallback_cost = weights.get((start, end), 10.0)  # Default high cost
    
    try:
        # Try quantum optimization approach
        qp = QuadraticProgram()
        nodes = list(G.nodes)
        
        # Create binary variables for each edge in the graph
        edge_vars = {}
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if G.has_edge(u, v):
                    var_name = f"x_{i}_{j}"
                    qp.binary_var(var_name)
                    edge_vars[(u, v)] = var_name
        
        # Objective function: minimize the total weight
        obj_linear = {}
        for (u, v), var_name in edge_vars.items():
            obj_linear[var_name] = weights.get((u, v), 0)
        qp.minimize(linear=obj_linear)
        
        # Constraints: flow conservation
        # For each node except start and end: incoming flow = outgoing flow
        for node in nodes:
            if node != start and node != end:
                flow_constraint = {}
                # Incoming edges
                for u in G.predecessors(node):
                    var_name = edge_vars.get((u, node), None)
                    if var_name:
                        flow_constraint[var_name] = 1
                # Outgoing edges
                for v in G.successors(node):
                    var_name = edge_vars.get((node, v), None)
                    if var_name:
                        flow_constraint[var_name] = -1
                
                if flow_constraint:  # Only add constraint if we have variables
                    qp.linear_constraint(flow_constraint, "==", 0)
        
        # Start node must have one outgoing edge
        start_constraint = {}
        for v in G.successors(start):
            var_name = edge_vars.get((start, v), None)
            if var_name:
                start_constraint[var_name] = 1
        
        if start_constraint:
            qp.linear_constraint(start_constraint, "==", 1)
        
        # End node must have one incoming edge
        end_constraint = {}
        for u in G.predecessors(end):
            var_name = edge_vars.get((u, end), None)
            if var_name:
                end_constraint[var_name] = 1
        
        if end_constraint:
            qp.linear_constraint(end_constraint, "==", 1)
        
        # Convert to QUBO and solve with QAOA
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        operator, offset = qubo.to_ising()
        
        # Set up QAOA with Sampler
        optimizer = SPSA(maxiter=50)
        sampler = Sampler()
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
        result = qaoa.compute_minimum_eigenvalue(operator)
        
        # Get the most probable bitstring
        if hasattr(result, 'eigenstate'):
            bitstring = max(result.eigenstate, key=result.eigenstate.get)
        elif hasattr(result, 'samples') and len(result.samples) > 0:
            # Alternative way to get result depending on Qiskit version
            bitstring = result.samples[0].x
        else:
            print("Could not get solution from QAOA result")
            return fallback_path, fallback_cost
        
        # Reconstruct the path
        selected_edges = []
        for (u, v), var_name in edge_vars.items():
            var_idx = list(qp.variables).index(qp.get_variable(var_name))
            if var_idx < len(bitstring) and bitstring[var_idx] == 1:
                selected_edges.append((u, v))
        
        # Construct the path from selected edges
        if not selected_edges:
            print("No edges selected in quantum solution")
            return fallback_path, fallback_cost
            
        # Try to construct a path using selected edges
        path = [start]
        current = start
        visited = set([start])
        max_iterations = len(nodes) * 2  # Avoid infinite loops
        iteration = 0
        
        while current != end and iteration < max_iterations:
            iteration += 1
            for u, v in selected_edges:
                if u == current and v not in visited:
                    path.append(v)
                    visited.add(v)
                    current = v
                    break
            else:
                # No valid next step found
                if iteration == 1:  # Failed on first step
                    print("Could not construct path from quantum solution")
                    return fallback_path, fallback_cost
                else:
                    # We started a path but couldn't complete it
                    # Try to find a way to the end node using shortest path
                    try:
                        completion_path = nx.shortest_path(G_weighted, source=current, target=end, weight='weight')
                        # Add all but the first node (we already have that)
                        path.extend(completion_path[1:])
                        break
                    except:
                        print("Could not complete partial quantum path")
                        return fallback_path, fallback_cost
        
        if current != end:
            print("Path construction did not reach the end node")
            return fallback_path, fallback_cost
            
        # Calculate the total cost using positive weights
        total_cost = sum(weights.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
        print(f"Quantum path: {path}, cost: {total_cost}")
        
        # If we have a valid path, return it
        if len(path) >= 2 and path[0] == start and path[-1] == end and total_cost > 0:
            return path, total_cost
        else:
            # Fallback to classical solution
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

        # Don't allow same start and end
        if start == end:
            return render_template("result.html", 
                                  amount=amount, 
                                  path="Error: Start and end nodes must be different", 
                                  total_cost=0.0)

        weights = ai_predict_weights(G)
        path, total_cost = quantum_route_optimization(G, weights, start, end)
        
        # Format path for display
        if path:
            path_display = " -> ".join(path)
        else:
            path_display = "No valid path found"
        
        return render_template("result.html", 
                              amount=amount, 
                              path=path_display, 
                              total_cost=total_cost)
    return render_template("index.html", nodes=nodes)

if __name__ == "__main__":
    app.run(debug=True)