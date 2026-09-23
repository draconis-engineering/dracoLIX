using Test
using DracoLIX

@testset "DracoLIX Julia interop" begin
    @test occursin("0.1", DracoLIX.version())

    A = [1.0 2.0; 3.0 4.0]
    B = [5.0 6.0; 7.0 8.0]
    C = DracoLIX.matmul(A, B)
    @test C ≈ [19.0 22.0; 43.0 50.0]

    # low-level API
    da = DracoLIX.from_matrix(A)
    db = DracoLIX.from_matrix(B)
    dc = DracoLIX.matmul(da, db)
    @test DracoLIX.to_matrix(dc) ≈ C
    @test size(dc) == (2,2)
end
