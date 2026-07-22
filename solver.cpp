#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>

extern "C" {

// Helper macros to map 3D/2D arrays to flat 1D pointers
#define GMEM(k, i, j) GMem_ptr[(k) * n * m + (i) * m + (j)]
#define VINPUT(i, k)  VInput_ptr[(i) * p + (k)]
#define VOUTPUT(j, k) VOutput_ptr[(j) * p + (k)]

// Helper functions for node indexing
inline int W_idx(int n, int m, int k, int i, int j) {
    return 2 * m * n * k + 2 * m * i + 2 * j;
}

inline int B_idx(int n, int m, int k, int i, int j) {
    return 2 * m * n * k + 2 * m * i + 2 * j + 1;
}

void solve_kcl(double* GMem_ptr, double* VInput_ptr, double* VOutput_ptr, 
               double* VKCL_ptr, int n, int m, int p, 
               double S, double L, double GW) {
    
    int num_nodes = 2 * n * m * p;
    Eigen::VectorXd IKCL = Eigen::VectorXd::Zero(num_nodes);
    std::vector<Eigen::Triplet<double>> triplets;
    
    // Pre-allocate space for triplets to avoid reallocation overhead
    // Roughly 7 non-zeros per row max (diagonal + up to 6 neighbors)
    triplets.reserve(num_nodes * 7);

    // =========================================================================
    // CASES 1-9: WORDLINE NODES W(i, j, k)
    // =========================================================================
    for (int k = 0; k < p; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                int row = W_idx(n, m, k, i, j);
                double G_mem = GMEM(k, i, j);
                double diag = G_mem;
                
                // Connection to corresponding Bitline
                triplets.push_back(Eigen::Triplet<double>(row, B_idx(n, m, k, i, j), -G_mem));

                // J-axis (Horizontal wire segments)
                if (j == 0) { // Cases 1, 2, 3 (Input boundary)
                    diag += S;
                    IKCL(row) += S * VINPUT(i, k);
                    if (m > 1) {
                        diag += GW;
                        triplets.push_back(Eigen::Triplet<double>(row, W_idx(n, m, k, i, j + 1), -GW));
                    }
                } else if (j == m - 1) { // Cases 7, 8, 9 (Right boundary)
                    diag += GW;
                    triplets.push_back(Eigen::Triplet<double>(row, W_idx(n, m, k, i, j - 1), -GW));
                } else { // Cases 4, 5, 6 (Internal J segments)
                    diag += 2 * GW;
                    triplets.push_back(Eigen::Triplet<double>(row, W_idx(n, m, k, i, j - 1), -GW));
                    triplets.push_back(Eigen::Triplet<double>(row, W_idx(n, m, k, i, j + 1), -GW));
                }

                // K-axis (Vertical vias)
                if (k == 0) { // Top layer
                    if (p > 1) {
                        diag += GW;
                        triplets.push_back(Eigen::Triplet<double>(row, W_idx(n, m, k + 1, i, j), -GW));
                    }
                } else if (k == p - 1) { // Bottom layer
                    diag += GW;
                    triplets.push_back(Eigen::Triplet<double>(row, W_idx(n, m, k - 1, i, j), -GW));
                } else { // Internal K segments
                    diag += 2 * GW;
                    triplets.push_back(Eigen::Triplet<double>(row, W_idx(n, m, k - 1, i, j), -GW));
                    triplets.push_back(Eigen::Triplet<double>(row, W_idx(n, m, k + 1, i, j), -GW));
                }

                // Main Diagonal
                triplets.push_back(Eigen::Triplet<double>(row, row, diag));
            }
        }
    }

    // =========================================================================
    // CASES 10-18: BITLINE NODES B(i, j, k)
    // =========================================================================
    for (int k = 0; k < p; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                int row = B_idx(n, m, k, i, j);
                double G_mem = GMEM(k, i, j);
                double diag = G_mem;
                
                // Connection to corresponding Wordline
                triplets.push_back(Eigen::Triplet<double>(row, W_idx(n, m, k, i, j), -G_mem));

                // I-axis (Vertical wire segments)
                if (i == 0) { // Cases 10, 11, 12 (Top boundary)
                    if (n > 1) {
                        diag += GW;
                        triplets.push_back(Eigen::Triplet<double>(row, B_idx(n, m, k, i + 1, j), -GW));
                    }
                } else if (i == n - 1) { // Cases 16, 17, 18 (Output boundary)
                    diag += L;
                    IKCL(row) += L * VOUTPUT(j, k);
                    diag += GW;
                    triplets.push_back(Eigen::Triplet<double>(row, B_idx(n, m, k, i - 1, j), -GW));
                } else { // Cases 13, 14, 15 (Internal I segments)
                    diag += 2 * GW;
                    triplets.push_back(Eigen::Triplet<double>(row, B_idx(n, m, k, i - 1, j), -GW));
                    triplets.push_back(Eigen::Triplet<double>(row, B_idx(n, m, k, i + 1, j), -GW));
                }

                // K-axis (Vertical vias)
                if (k == 0) { // Top layer
                    if (p > 1) {
                        diag += GW;
                        triplets.push_back(Eigen::Triplet<double>(row, B_idx(n, m, k + 1, i, j), -GW));
                    }
                } else if (k == p - 1) { // Bottom layer
                    diag += GW;
                    triplets.push_back(Eigen::Triplet<double>(row, B_idx(n, m, k - 1, i, j), -GW));
                } else { // Internal K segments
                    diag += 2 * GW;
                    triplets.push_back(Eigen::Triplet<double>(row, B_idx(n, m, k - 1, i, j), -GW));
                    triplets.push_back(Eigen::Triplet<double>(row, B_idx(n, m, k + 1, i, j), -GW));
                }

                // Main Diagonal
                triplets.push_back(Eigen::Triplet<double>(row, row, diag));
            }
        }
    }

    // =========================================================================
    // MATRIX COMPRESSION & SOLVING
    // =========================================================================
    Eigen::SparseMatrix<double> GKCL(num_nodes, num_nodes);
    GKCL.setFromTriplets(triplets.begin(), triplets.end());

    // Symmetric Positive-Definite (SPD) solver
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(GKCL);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed. Is the matrix truly SPD?" << std::endl;
        return;
    }

    Eigen::VectorXd VKCL = solver.solve(IKCL);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Solving failed." << std::endl;
        return;
    }

    // Map output to Python pointer
    for (int idx = 0; idx < num_nodes; ++idx) {
        VKCL_ptr[idx] = VKCL(idx);
    }
}

} // extern "C"