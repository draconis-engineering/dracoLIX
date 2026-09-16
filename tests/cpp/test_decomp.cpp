#include "../../core/include/dracolix/dracolix.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace dracolix;
using namespace dracolix::decomp;

void test_lu_solve_det_inverse() {
    Array<double> A({2,2}); A.at(0,0)=4; A.at(0,1)=3; A.at(1,0)=6; A.at(1,1)=3;
    auto f = lu(A);
    // det
    assert(std::abs(determinant(A) + 6) < 1e-9);
    // solve
    Array<double> b({2}); b[0]=1; b[1]=1;
    auto x = solve(A,b);
    auto Ax = linalg::matvec(A,x);
    assert(std::abs(Ax[0]-1)<1e-9 && std::abs(Ax[1]-1)<1e-9);
    // inverse
    auto inv = inverse(A);
    auto eye = linalg::matmul(A, inv);
    for(size_t i=0;i<2;++i) for(size_t j=0;j<2;++j) assert(std::abs(eye.at(i,j) - (i==j?1:0))<1e-9);
    // rank
    assert(rank(A)==2);
    Array<double> S({2,2}); S.at(0,0)=1; S.at(0,1)=2; S.at(1,0)=2; S.at(1,1)=4;
    assert(rank(S)==1);
    std::cout << " lu/solve/det/inv/rank ok\n";
}

void test_solve_multi_rhs() {
    Array<double> A({2,2}); A.at(0,0)=2; A.at(0,1)=1; A.at(1,0)=1; A.at(1,1)=3;
    Array<double> B({2,2}); B.at(0,0)=1; B.at(0,1)=2; B.at(1,0)=3; B.at(1,1)=4;
    auto X = solve(A,B);
    auto AX = linalg::matmul(A,X);
    for(size_t i=0;i<2;++i) for(size_t j=0;j<2;++j) assert(std::abs(AX.at(i,j)-B.at(i,j))<1e-9);
    std::cout << " solve multi-rhs ok\n";
}

int main(){ test_lu_solve_det_inverse(); test_solve_multi_rhs(); std::cout<<"All decomp tests passed\n"; return 0; }
