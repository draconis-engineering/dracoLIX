#include "../../core/include/dracolix/dracolix.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
using namespace dracolix;
using namespace dracolix::linalg;

void test_matmul() {
    Array<double> A({2,3}); A.at(0,0)=1; A.at(0,1)=2; A.at(0,2)=3; A.at(1,0)=4; A.at(1,1)=5; A.at(1,2)=6;
    Array<double> B({3,2}); B.at(0,0)=7; B.at(0,1)=8; B.at(1,0)=9; B.at(1,1)=10; B.at(2,0)=11; B.at(2,1)=12;
    auto C = matmul(A,B);
    assert(C.shape()[0]==2 && C.shape()[1]==2);
    assert(C.at(0,0)==58 && C.at(0,1)==64 && C.at(1,0)==139 && C.at(1,1)==154);
    std::cout<<" matmul ok\n";
}
void test_matvec() {
    Array<double> A({2,2}); A.at(0,0)=1; A.at(0,1)=2; A.at(1,0)=3; A.at(1,1)=4;
    Array<double> x({2}); x[0]=5; x[1]=6;
    auto y = matvec(A,x);
    assert(y[0]==17 && y[1]==39);
    std::cout<<" matvec ok\n";
}
void test_dot_norm() {
    Array<double> a({3}); a[0]=1; a[1]=2; a[2]=3;
    Array<double> b({3}); b[0]=4; b[1]=5; b[2]=6;
    assert(dot(a,b)==32);
    assert(std::abs(norm(a,2)-std::sqrt(14))<1e-9);
    assert(norm(a,1)==6);
    assert(norm(a,0)==3);
    std::cout<<" dot/norm ok\n";
}
void test_diag() {
    Array<double> A({3,3}); for(size_t i=0;i<9;++i) A[i]=i;
    auto d = diagonal(A); assert(d[0]==0 && d[1]==4 && d[2]==8);
    auto M = diag(d); assert(M.at(1,1)==4 && M.at(0,1)==0);
    std::cout<<" diag ok\n";
}
void test_batched() {
    Array<double> A({2,2,3}); Array<double> B({2,3,2});
    for(size_t i=0;i<A.size();++i) A[i]=i+1;
    for(size_t i=0;i<B.size();++i) B[i]=i+1;
    auto C = matmul(A,B);
    assert((C.shape()==std::vector<size_t>{2,2,2}));
    assert(C.at(std::vector<size_t>{0,0,0})==22);
    assert(C.at(std::vector<size_t>{0,1,0})==49);
    // broadcast batch
    Array<double> A2({2,1,3}); for(size_t i=0;i<A2.size();++i) A2[i]=1;
    Array<double> B2({3,2}); for(size_t i=0;i<B2.size();++i) B2[i]=i+1;
    auto C2 = matmul(A2,B2);
    assert((C2.shape()==std::vector<size_t>{2,1,2}));
    // matvec batched
    Array<double> M({2,2,3}); for(size_t i=0;i<M.size();++i) M[i]=i+1;
    Array<double> x({3}); x[0]=1; x[1]=1; x[2]=1;
    auto y = matvec(M,x);
    assert((y.shape()==std::vector<size_t>{2,2}));
    // 4D
    Array<double> A4({2,2,2,3}); for(size_t i=0;i<A4.size();++i) A4[i]=1;
    Array<double> B4({2,2,3,2}); for(size_t i=0;i<B4.size();++i) B4[i]=1;
    auto C4 = matmul(A4,B4);
    assert((C4.shape()==std::vector<size_t>{2,2,2,2}));
    std::cout<<" batched ok\n";
}
int main(){ test_matmul(); test_matvec(); test_dot_norm(); test_diag(); test_batched(); std::cout<<"All linalg tests passed\n"; return 0; }
