
// Optimized inner loop ordering (i, k, j) instead of (i, j, k)
for (int i = 0; i < R1; ++i) {
    for (int k = 0; k < C1; ++k) {
        for (int j = 0; j < C2; ++j) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
