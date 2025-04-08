
import networkx as nx
import numpy as np
import tensorflow as tf
from qiskit_aer import AerSimulator
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from flask import Flask, request, render_template, jsonify
import logging
from typing import List, Dict, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Create a simple payment network (undirected graph)
G = nx.Graph()
banks = ["A", "B", "C", "D", "E"]
G.add_nodes_from(banks)
edges = [("A", "B", {"latency": 10, "fee": 2, "liquidity": 10000, "forex_rate": 0.5}),
         ("A", "C", {"latency": 15, "fee": 3, "liquidity": 8000, "forex_rate": 0.7}),
         ("B", "D", {"latency": 20, "fee": 1, "liquidity": 12000, "forex_rate": 0.3}),
         ("C", "D", {"latency": 12, "fee": 4, "liquidity": 6000, "forex_rate": 0.6}),
         ("D", "E", {"latency": 18, "fee": 2, "liquidity": 9000, "forex_rate": 0.4}),
         ("B", "E", {"latency": 25, "fee": 3, "liquidity": 7000, "forex_rate": 0.8})]
G.add_edges_from(edges)

# Simulated node status (True = up, False = down)
node_status = {bank: True for bank in banks}
# For demonstration, assume bank "D" is down
node_status["D"] = False

# Step 2: Enhanced AI model to predict optimal paths
class EnhancedRoutingModel:
    def __init__(self, feature_count: int = 6):
        self.feature_count = feature_count
        self.model = self._build_model()
        self.history = None
        
    def _build_model(self) -> tf.keras.Model:
        """Build a more complex neural network with dropout for regularization"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(self.feature_count,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 8) -> None:
        """Train the model with early stopping"""
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=15,
            restore_best_weights=True
        )
        
        # Learning rate scheduler for better convergence
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stopping, lr_scheduler]
        )
        
        logger.info(f"Model trained for {len(self.history.history['loss'])} epochs")
        logger.info(f"Final loss: {self.history.history['loss'][-1]:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the model"""
        return self.model.predict(X, verbose=0)
    
    def save(self, filepath: str) -> None:
        """Save the model to disk"""
        self.model.save(filepath)
    
    def load(self, filepath: str) -> None:
        """Load the model from disk"""
        self.model = tf.keras.models.load_model(filepath)

def train_ai_model() -> EnhancedRoutingModel:
    """Train an enhanced AI model with more features and data"""
    # Features: [start_idx, end_idx, total_latency, total_fee, min_liquidity, total_forex_cost]
    X = np.array([
        [0, 1, 10, 2, 10000, 0.5],
        [0, 2, 15, 3, 8000, 0.7],
        [1, 3, 20, 1, 12000, 0.3],
        [2, 3, 12, 4, 6000, 0.6],
        [3, 4, 18, 2, 9000, 0.4],
        [1, 4, 25, 3, 7000, 0.8],
        [0, 3, 30, 3, 10000, 0.8],
        [0, 4, 45, 5, 7000, 1.3],
        [1, 2, 35, 4, 8000, 0.9],
        # Additional synthetic data points
        [0, 1, 12, 3, 9500, 0.6],
        [0, 2, 14, 2, 8500, 0.8],
        [1, 3, 22, 2, 11000, 0.4],
        [2, 3, 10, 3, 7000, 0.5],
        [3, 4, 20, 3, 8500, 0.5],
        [1, 4, 28, 4, 6500, 0.9],
        [0, 3, 32, 4, 9500, 0.7],
        [0, 4, 42, 6, 7500, 1.2],
        [1, 2, 33, 3, 8500, 0.8],
    ])
    
    # Lower score = better path (enhanced scoring)
    y = np.array([
        0.15, 0.25, 0.1, 0.3, 0.2, 0.35, 0.4, 0.6, 0.45,
        0.18, 0.22, 0.12, 0.28, 0.22, 0.38, 0.42, 0.55, 0.42
    ])
    
    # Apply feature normalization
    X_norm = X.copy()
    X_norm[:, 2] = X_norm[:, 2] / 50.0  # Normalize latency
    X_norm[:, 3] = X_norm[:, 3] / 10.0  # Normalize fee
    X_norm[:, 4] = X_norm[:, 4] / 15000.0  # Normalize liquidity
    X_norm[:, 5] = X_norm[:, 5] / 2.0  # Normalize forex cost
    
    model = EnhancedRoutingModel(feature_count=6)
    model.train(X_norm, y, epochs=150, batch_size=8)
    return model

ai_model = train_ai_model()

def extract_path_features(path: List[str], graph: nx.Graph, amount: float = 1000.0) -> Tuple[float, float, float, float]:
    """Extract features from a path in the network"""
    total_latency = 0
    total_fee = 0
    min_liquidity = float('inf')
    total_forex_cost = 0
    
    # Calculate how much of the original amount reaches each node
    remaining_amount = amount
    
    for i in range(len(path) - 1):
        edge = graph[path[i]][path[i+1]]
        total_latency += edge["latency"]
        total_fee += edge["fee"]
        min_liquidity = min(min_liquidity, edge["liquidity"])
        
        # Calculate forex cost as a percentage of the remaining amount
        forex_cost = remaining_amount * (edge["forex_rate"] / 100.0)
        total_forex_cost += forex_cost
        
        # Reduce the remaining amount by the forex cost
        remaining_amount -= forex_cost
    
    return total_latency, total_fee, min_liquidity, total_forex_cost

def ai_predict(start: str, end: str, amount: float = 1000.0, graph: nx.Graph = G, max_hops: int = 3) -> Optional[List[str]]:
    """Use the AI model to predict the best path"""
    paths = list(nx.all_simple_paths(graph, start, end, cutoff=max_hops))
    
    if not paths:
        logger.warning(f"No paths found between {start} and {end}")
        return None
    
    logger.info(f"Found {len(paths)} paths between {start} and {end}")
    
    inputs = []
    for path in paths:
        total_latency, total_fee, min_liquidity, total_forex_cost = extract_path_features(path, graph, amount)
        
        # Create feature vector: [start_idx, end_idx, total_latency, total_fee, min_liquidity, total_forex_cost]
        inputs.append([
            banks.index(start),
            banks.index(end),
            total_latency,
            total_fee,
            min_liquidity,
            total_forex_cost
        ])
    
    # Apply normalization similar to training data
    inputs_norm = np.array(inputs)
    inputs_norm[:, 2] = inputs_norm[:, 2] / 50.0  # Normalize latency
    inputs_norm[:, 3] = inputs_norm[:, 3] / 10.0  # Normalize fee
    inputs_norm[:, 4] = inputs_norm[:, 4] / 15000.0  # Normalize liquidity
    inputs_norm[:, 5] = inputs_norm[:, 5] / 2.0  # Normalize forex cost
    
    # Make predictions
    scores = ai_model.predict(inputs_norm)
    best_idx = np.argmin(scores)
    
    logger.info(f"AI model selected path: {' -> '.join(paths[best_idx])}")
    return paths[best_idx]

# Step 3: Enhanced quantum optimization
class QuantumRouteOptimizer:
    """Enhanced quantum optimization for payment routing"""
    
    def __init__(self, backend_name: str = 'statevector_simulator'):
        self.backend = AerSimulator(method='statevector')
        self.sampler = Sampler()
    
    def _create_quantum_program(self, paths: List[List[str]], graph: nx.Graph, amount: float) -> Tuple[QuadraticProgram, Dict[int, float]]:
        """Create a quantum program for path optimization"""
        qp = QuadraticProgram("payment_routing")
        
        # Add binary variables for each path
        for i in range(len(paths)):
            qp.binary_var(f"x_{i}")
        
        # Calculate costs for each path with weighted factors
        linear_costs = {}
        path_costs = {}
        
        for i, path in enumerate(paths):
            total_latency, total_fee, min_liquidity, total_forex_cost = extract_path_features(path, graph, amount)
            
            # Create a weighted cost function combining all factors
            # Lower values are better
            weighted_cost = (
                0.30 * (total_latency / 10) +  # Normalize and weight latency
                0.20 * total_fee +             # Weight fee importance
                0.20 * (1000000 / min_liquidity) +  # Invert liquidity so lower is better
                0.30 * total_forex_cost        # Weight forex cost (already as percentage)
            )
            
            linear_costs[f"x_{i}"] = weighted_cost
            path_costs[i] = weighted_cost
            
            print(f"Path {i}: {' -> '.join(path)}, Cost: {weighted_cost:.4f}")
        
        # Set the objective to minimize the cost
        qp.minimize(linear=linear_costs)
        
        # Add constraint: exactly one path must be selected
        qp.linear_constraint(linear={f"x_{i}": 1 for i in range(len(paths))}, sense="==", rhs=1)
        
        return qp, path_costs
    
    def optimize(self, start: str, end: str, ai_path: List[str], amount: float, graph: nx.Graph = G, 
                max_hops: int = 3, use_qaoa: bool = True) -> List[Tuple[List[str], float]]:
        """Run quantum optimization on possible paths"""
        try:
            paths = list(nx.all_simple_paths(graph, start, end, cutoff=max_hops))
            
            if not paths or len(paths) < 2:
                print(f"Quantum skipped: Only {len(paths)} path(s) found")
                if paths:
                    cost = self._calculate_path_cost(paths[0], graph, amount)
                    return [(paths[0], cost)]
                else:
                    return [(ai_path, self._calculate_path_cost(ai_path, graph, amount))]
        
            print(f"Quantum running: Found {len(paths)} paths to optimize")
            for i, path in enumerate(paths):
                print(f"Path {i}: {' -> '.join(path)}")
            
            # Create quantum program
            qp, path_costs = self._create_quantum_program(paths, graph, amount)
            
            if use_qaoa:
                # Use QAOA with better parameters
                optimizer = COBYLA(maxiter=100)  # COBYLA can work better than SPSA in some cases
                qaoa = QAOA(sampler=self.sampler, optimizer=optimizer, reps=2)  # Increased reps
                quantum_optimizer = MinimumEigenOptimizer(qaoa)
                print("Running QAOA optimizer with 2 repetitions")
            else:
                # Fallback to classical solver for comparison
                classical_solver = NumPyMinimumEigensolver()
                quantum_optimizer = MinimumEigenOptimizer(classical_solver)
                print("Running classical optimizer (NumPy)")
            
            result = quantum_optimizer.solve(qp)
            
            # Process results
            selected_indices = [i for i, val in enumerate(result.x) if val > 0.5]
            if selected_indices:
                selected_idx = selected_indices[0]
                print(f"Quantum chose Path {selected_idx}: {' -> '.join(paths[selected_idx])}")
            else:
                print("Quantum optimization did not select any path clearly")
                # Fallback to minimum cost path
                selected_idx = min(path_costs.items(), key=lambda x: x[1])[0]
            
            # Return all paths sorted by cost
            all_paths = [(paths[i], path_costs[i]) for i in range(len(paths))]
            all_paths.sort(key=lambda x: x[1])  # Sort by cost
            
            return all_paths
            
        except Exception as e:
            print(f"Quantum optimization error: {str(e)}")
            return [(ai_path, self._calculate_path_cost(ai_path, graph, amount))]
    
    def _calculate_path_cost(self, path: List[str], graph: nx.Graph, amount: float) -> float:
        """Calculate the cost of a path for display"""
        total_latency, total_fee, _, total_forex_cost = extract_path_features(path, graph, amount)
        # Include forex cost in the total path cost
        return total_latency + total_fee + total_forex_cost

# Create quantum optimizer
quantum_optimizer = QuantumRouteOptimizer()

# Helper function to calculate path cost for display
def calculate_path_cost(path, amount):
    total_cost = 0
    forex_cost = 0
    
    # Calculate the base cost (latency + fee)
    base_cost = sum(G[path[i]][path[i+1]]["latency"] + G[path[i]][path[i+1]]["fee"]
                   for i in range(len(path) - 1))
    
    # Calculate forex cost
    remaining_amount = amount
    for i in range(len(path) - 1):
        edge = G[path[i]][path[i+1]]
        forex_percentage = edge["forex_rate"] / 100.0
        current_forex_cost = remaining_amount * forex_percentage
        forex_cost += current_forex_cost
        remaining_amount -= current_forex_cost
    
    total_cost = base_cost + forex_cost
    return total_cost

# Flask web interface
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    nodes = banks
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
        
        # Check if start and end are the same
        if start == end:
            return render_template("result.html",
                                  path="Error: Start and end banks cannot be the same!",
                                  amount=amount,
                                  total_cost=0)
        
        ai_path = ai_predict(start, end, amount)
        if not ai_path:
            return render_template("result.html", 
                                  path="Error: No path found between banks!", 
                                  amount=amount,
                                  total_cost=0)
        
        # Run quantum optimization
        all_paths = quantum_optimizer.optimize(start, end, ai_path, amount)
        
        # Find the first viable path considering node status AND liquidity
        final_path = None
        total_cost = 0
        original_path_msg = ""
        liquidity_issues = []
        
        for idx, (path, cost) in enumerate(all_paths):
            # Check for offline nodes
            nodes_down = [node for node in path if not node_status[node]]
            
            # Check for liquidity constraints
            liquidity_constraint = False
            insufficient_edge = None
            
            # Verify sufficient liquidity at each hop
            for i in range(len(path) - 1):
                edge = G[path[i]][path[i+1]]
                if amount > edge["liquidity"]:
                    liquidity_constraint = True
                    insufficient_edge = f"{path[i]} -> {path[i+1]}"
                    break
            
            # Path is viable if all nodes are up AND there's sufficient liquidity
            if not nodes_down and not liquidity_constraint:
                final_path = path
                total_cost = calculate_path_cost(path, amount)
                if idx == 0:
                    path_display = f"Optimal path: {' -> '.join(final_path)}"
                else:
                    path_display = f"Alternative path #{idx+1}: {' -> '.join(final_path)}"
                break
            elif idx == 0:
                # If the optimal path has issues, note them
                if nodes_down:
                    original_path_msg = (f"Optimal path {' -> '.join(path)} was rejected "
                                        f"due to node(s) {', '.join(nodes_down)} being down.")
                if liquidity_constraint:
                    liquidity_issues.append(f"Optimal path {' -> '.join(path)} was rejected "
                                          f"due to insufficient liquidity at {insufficient_edge} "
                                          f"(required: {amount}, available: {G[path[i]][path[i+1]]['liquidity']}).")
            else:
                # Track liquidity issues for all paths
                if liquidity_constraint:
                    liquidity_issues.append(f"Path #{idx+1} {' -> '.join(path)} has insufficient liquidity at "
                                          f"{insufficient_edge} (required: {amount}, available: {G[path[i]][path[i+1]]['liquidity']}).")

        if not final_path:
            error_msg = "Error: No viable path found. "
            if liquidity_issues:
                error_msg += f"Liquidity issues: {' '.join(liquidity_issues)}"
            if original_path_msg:
                error_msg += f" Node issues: {original_path_msg}"
                
            return render_template("result.html", 
                                  path=error_msg, 
                                  amount=amount,
                                  total_cost=0)
        
        # Combine messages if the original path was rejected
        final_display = ""
        if original_path_msg or liquidity_issues:
            if original_path_msg:
                final_display += f"{original_path_msg}<br>"
            if liquidity_issues:
                final_display += f"{liquidity_issues[0]}<br>"
            final_display += path_display
        else:
            final_display = path_display
        
        return render_template("result.html", 
                              amount=amount, 
                              path=final_display, 
                              total_cost=total_cost)
                              
    except Exception as e:
        return render_template("result.html", 
                              path=f"Error: {str(e)}", 
                              amount=amount if 'amount' in locals() else 0,
                              total_cost=0)

@app.route("/network_status", methods=["GET"])
def network_status():
    """Return the current network status as JSON"""
    nodes_data = []
    for node in banks:
        nodes_data.append({
            "name": node,
            "status": "online" if node_status[node] else "offline"
        })
    
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            "source": u,
            "target": v,
            "latency": data["latency"],
            "fee": data["fee"],
            "liquidity": data["liquidity"],
            "forex_rate": data.get("forex_rate", 0.5)  # Default if not present
        })
    
    return jsonify({
        "nodes": nodes_data,
        "edges": edges_data
    })

if __name__ == "__main__":
    app.run(debug=True)