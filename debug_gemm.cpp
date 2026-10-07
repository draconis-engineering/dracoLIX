#include "dracolix/cpu.hpp"
#include "dracolix/kernels/dispatch.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// Your exact random fill logic
template <typename T> void fill_rand(std::vector<T> &v, uint64_t seed) {
	for (size_t i = 0; i < v.size(); ++i) {
		seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
		v[i] = static_cast<T>((seed >> 33) % 100000) / T(100000) - T(0.5);
	}
}

void test_shape(size_t M, size_t N, size_t P) {
	std::cout << "\n===========================================\n";
	std::cout << " TESTING SHAPE: " << M << "x" << N << " * " << N << "x" << P
			  << "\n";
	std::cout << "===========================================\n";

	std::vector<float> A(M * N), B(N * P);
	std::vector<float> C_ref(M * P, 0.0f), C_opt(M * P, 0.0f);

	fill_rand(A, 0x9e3779b97f4a7c15ULL ^ M);
	fill_rand(B, 0xc2b2ae3d27d4eb4fULL ^ P);

	// Reference implementation
	for (size_t i = 0; i < M; ++i) {
		for (size_t k = 0; k < N; ++k) {
			float aik = A[i * N + k];
			for (size_t j = 0; j < P; ++j) {
				C_ref[i * P + j] += aik * B[k * P + j];
			}
		}
	}

	// Optimized execution
	dracolix::kernels::dispatch_gemm_f32(A.data(), B.data(), C_opt.data(), M, N,
										 P);

	// Print the first few results or any broken results
	std::cout << "Index | Reference | Optimized | Difference\n";
	std::cout << "-------------------------------------------\n";
	size_t printed = 0;
	for (size_t i = 0; i < M * P; ++i) {
		float diff = std::fabs(C_ref[i] - C_opt[i]);
		if (diff > 1e-4 || printed < 10) {
			std::cout << i << "     | " << C_ref[i] << "    | " << C_opt[i]
					  << "    | " << diff;
			if (diff > 1e-4)
				std::cout << "  <-- BUG HERE!";
			std::cout << "\n";
			printed++;
		}
		if (printed >= 25) {
			std::cout << "... truncation for readability ...\n";
			break;
		}
	}
}
int main() {
	// This will force the multi-threaded block/tiling code to activate!
	test_shape(47, 47, 47);
	return 0;
}
