# README.md

## :rocket: 1. Solution Name

**AI Enhanced Quantum Payment Router**

## :mag: 2. Solution Description

The **AI Enhanced Quantum Payment Router** is an innovative solution designed to optimize cross-border payment routing in financial networks. Cross-border payments often face challenges such as high transaction fees, variable forex rates, network latency, and liquidity constraints, leading to inefficiencies and increased costs for financial institutions and their clients. Traditional routing methods may not dynamically account for real-time network conditions or optimize across multiple factors simultaneously.

Our solution addresses these issues by leveraging **artificial intelligence (AI)** and **quantum computing principles** to identify the most cost-effective and efficient payment paths between banks. It considers factors like latency, fees, liquidity, and forex rates, while also adapting to real-time network conditions (e.g., offline nodes or insufficient liquidity). The goal is to minimize transaction costs, reduce delays, and ensure reliable payment delivery, ultimately enhancing the efficiency of global financial transactions.

## :star: 3. Solution Features

- **AI-Powered Path Prediction**: Uses a deep neural network to predict optimal payment paths based on features like latency, fees, liquidity, and forex costs.
- **Quantum Optimization**: Employs quantum-inspired algorithms (QAOA) to refine path selection, balancing multiple weighted factors for a globally optimal solution.
- **Dynamic Network Awareness**: Accounts for real-time node status (online/offline) and liquidity constraints, rerouting payments if necessary.
- **User-Friendly Interface**: A Flask-based web application with an intuitive front-end for users to input payment details (start node, end node, amount) and view optimized routes.
- **Comprehensive Cost Calculation**: Incorporates latency, fees, and forex costs into a total cost metric, ensuring transparency and accuracy.
- **Path Visualization**: Displays the selected payment path with a visual representation in the results page.
- **Error Handling**: Provides detailed feedback on issues like offline nodes, insufficient liquidity, or invalid inputs.
- **Scalability**: Designed to handle a network of banks (e.g., A, B, C, D, E) with extensible graph-based architecture.

## :gear: 4. Technologies and Architecture Used

### Technologies:
- **Python**: Core programming language for backend logic, AI modeling, and quantum optimization.
- **TensorFlow**: For building and training the AI neural network model.
- **Qiskit**: Quantum computing framework for implementing QAOA and optimization algorithms.
- **NetworkX**: Library for creating and analyzing the payment network graph.
- **Flask**: Lightweight web framework for the user interface and API endpoints.
- **HTML/CSS/JavaScript**: For front-end development with responsive design and interactive elements.
- **NumPy**: For numerical computations and data preprocessing.
- **AerSimulator**: Qiskit’s quantum simulator for running quantum algorithms.

### Architecture:
- **Graph-Based Network**: Represents banks as nodes and payment channels as edges with attributes (latency, fee, liquidity, forex_rate).
- **AI Model**: A multi-layer neural network with batch normalization and dropout layers, trained on synthetic data to score payment paths.
- **Quantum Optimizer**: Uses a Quadratic Program to formulate the routing problem, solved with QAOA or a classical fallback (NumPyMinimumEigensolver).
- **Web Interface**: Flask app with two main routes:
  - `/` (GET): Displays the input form.
  - `/route` (POST): Processes inputs, runs AI and quantum optimization, and renders results.
  - `/network_status` (GET): Provides JSON data on network status for potential extensions.
- **Modular Design**: Separates AI training, path feature extraction, quantum optimization, and web logic into distinct components for maintainability.

## :dart: 5. Design Purpose

The code is designed to:
- Simulate a payment network with banks (A, B, C, D, E) and predefined connections.
- Train an AI model to predict optimal payment paths based on historical and synthetic data.
- Use quantum optimization to refine path selection, ensuring the best trade-off between latency, fees, liquidity, and forex costs.
- Provide a web-based tool for users to input a payment amount and source/destination banks, then receive an optimized route with total cost.
- Handle real-world constraints like offline nodes (e.g., bank D is down) and insufficient liquidity, offering alternative paths when needed.
- Serve as a proof-of-concept for integrating AI and quantum computing in financial routing systems.

## :computer: 6. Languages Used

- **Python 3.x**: Backend logic, AI model, quantum optimization, and Flask app.
- **HTML**: Templates (`index.html` for input, `result.html` for output).
- **CSS**: Custom styling with FIS branding colors and animations (quantum particles, pulsing nodes).
- **JavaScript**: Client-side validation (e.g., preventing same start/end nodes) and DOM manipulation.
- **Jinja2**: Templating engine for Flask to dynamically render HTML pages.

## :package: 7. Open Source or Proprietary Software Used

### Open Source Software:
- **TensorFlow**: Open-source machine learning framework (Apache 2.0 License).
- **Qiskit**: Open-source quantum computing framework (Apache 2.0 License).
- **NetworkX**: Open-source graph library (BSD License).
- **Flask**: Open-source web framework (BSD License).
- **NumPy**: Open-source numerical computing library (BSD License).
- **Font Awesome**: Open-source icon library (MIT License for CSS/JS, SIL OFL for fonts).
- **AerSimulator**: Part of Qiskit, open-source (Apache 2.0 License).

### Proprietary Software:
- None explicitly used in this project. The solution is built entirely on open-source tools.

## :star: 8. Additional Information

- **Innovation**: This project uniquely combines AI and quantum computing, showcasing a forward-thinking approach to financial optimization. While it uses a quantum simulator (AerSimulator) due to hardware limitations, it’s designed to scale with real quantum hardware in the future.
- **Extensibility**: The graph-based network can be expanded with more nodes and edges, and the AI model can be retrained with real-world data for production use.
- **Realism**: Incorporates practical constraints like node downtime (e.g., bank D offline) and liquidity checks, reflecting real-world banking scenarios.
- **Limitations**: 
  - Currently uses synthetic data for AI training; real transaction data would improve accuracy.
  - Quantum optimization is simulated, not run on actual quantum hardware.
  - Assumes a small, static network for demonstration; scaling to larger networks may require optimization.
- **Setup Instructions**: 
1. Install dependencies:  
     - `pip install networkx`  
     - `pip install numpy`  
     - `pip install flask`  
     - `pip install qiskit==1.2.0`  
     - `pip install qiskit-aer==0.17.0`  
     - `pip install qiskit-optimization==0.6.1`  
     - `pip install qiskit-algorithms==0.3.1`  
     - `pip install tensorflow`
  3. Run the Flask app: `python payment_router.py`.
  4. Access the web interface at `http://localhost:5000`.
- **Future Enhancements**: 
  - Integrate real-time banking APIs for dynamic network data.
  - Deploy on a cloud platform with a real quantum backend (e.g., IBM Quantum).
  - Add multi-currency support and advanced forex rate modeling.
- **Demo Output**:  
  - **Input Form**:  
    ![Input Form](https://drive.google.com/file/d/1IpyuaOwjncsGuKfS9qJZfEiWtsb65mxu/view)  
  - **Result Page**:  
    ![Result Page](https://via.placeholder.com/600x400.png?text=Result+Page+Image)   
---

© 2025 FIS Global. Submitted by [DevSquad] for INNOVATE48.
