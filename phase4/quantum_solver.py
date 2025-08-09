# quantum_solver.py
import numpy as np
import pennylane as qml
import torch 
import time

def create_ising_hamiltonian(Q_matrix):
    """Creates J and h coefficients from the Q matrix."""
    N = Q_matrix.shape[0]
    J = np.zeros((N, N))
    h = np.zeros(N)
    for n in range(N):
        for m in range(n + 1, N):
            J[n, m] = -2 * np.real(Q_matrix[n, m])
    return J, h

def run_qaoa_optimization(Q_matrix, params):
    N = params["N"]
    p_layers = params["qaoa_p_layers"]
    n_steps = params["qaoa_n_steps"]
    lr = params["qaoa_lr"]
    N_qaoa = params["qaoa_sim_qubits"]

    print(f"\nWARNING: System N={N} is too large. Simulating QAOA for the first {N_qaoa} qubits.")
    
    J, h = create_ising_hamiltonian(Q_matrix)
    J_qaoa, h_qaoa = J[:N_qaoa, :N_qaoa], h[:N_qaoa]

    print(f"Setting up QAOA with p={p_layers} layers for {N_qaoa} qubits...")
    dev = qml.device("default.qubit", wires=N_qaoa)
    
    cost_coeffs = [J_qaoa[i,j] for i in range(N_qaoa) for j in range(i+1, N_qaoa) if J_qaoa[i,j] != 0]
    cost_obs = [qml.PauliZ(i) @ qml.PauliZ(j) for i in range(N_qaoa) for j in range(i+1, N_qaoa) if J_qaoa[i,j] != 0]
    cost_h = qml.Hamiltonian(cost_coeffs, cost_obs)

    mixer_coeffs = [1.0] * N_qaoa
    mixer_obs = [qml.PauliX(i) for i in range(N_qaoa)]
    mixer_h = qml.Hamiltonian(mixer_coeffs, mixer_obs)
    
    def qaoa_circuit_manual(params, **kwargs):
        gammas, betas = params[0], params[1]
        for i in range(N_qaoa): qml.Hadamard(wires=i)
        for p in range(p_layers):
            qml.qaoa.cost_layer(gammas[p], cost_h)
            qml.qaoa.mixer_layer(betas[p], mixer_h)

    @qml.qnode(dev, interface="torch")
    def cost_function(params):
        qaoa_circuit_manual(params)
        return qml.expval(cost_h)

    # ... Optimization loop ...
    params_init = (2*np.pi*np.random.rand(p_layers), 2*np.pi*np.random.rand(p_layers))
    params_torch = [torch.tensor(p, requires_grad=True) for p in params_init]
    optimizer = torch.optim.Adam(params_torch, lr=lr)
    
    print("Starting QAOA optimization...")
    start_time = time.time()
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = cost_function(params_torch)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    print(f"QAOA optimization finished in {end_time - start_time:.2f} seconds.")

    # ... Extract solution ...
    @qml.qnode(dev)
    def probability_circuit(params):
        qaoa_circuit_manual(params)
        return qml.probs(wires=range(N_qaoa))
    final_params = (params_torch[0].detach().numpy(), params_torch[1].detach().numpy())
    probs = probability_circuit(final_params)
    v_qaoa_small = np.array([1 if bit == '0' else -1 for bit in format(np.argmax(probs), f'0{N_qaoa}b')])
    
    v_qaoa_full = np.ones(N)
    v_qaoa_full[:N_qaoa] = v_qaoa_small
    return v_qaoa_full
