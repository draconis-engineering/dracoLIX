#pragma once
// Numerical methods — Phase 6 prototype
// Root finding, interpolation, differentiation, integration, ODE (Euler/RK4/Adaptive)
// Licensed under GPL-3.0-only
#include "array.hpp"
#include <functional>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace dracolix::numerics {

// ---------- Root finding ----------
inline double bisect(std::function<double(double)> f, double a, double b, double tol=1e-10, int max_iter=1000){
    double fa=f(a), fb=f(b);
    if(fa*fb>0) throw std::invalid_argument("bisect: f(a) and f(b) must bracket root");
    for(int i=0;i<max_iter;++i){
        double m=0.5*(a+b);
        double fm=f(m);
        if(std::abs(fm)<tol || (b-a)<tol) return m;
        if(fa*fm<=0){ b=m; fb=fm; } else { a=m; fa=fm; }
    }
    return 0.5*(a+b);
}
inline double newton(std::function<double(double)> f, std::function<double(double)> df, double x0, double tol=1e-10, int max_iter=100){
    double x=x0;
    for(int i=0;i<max_iter;++i){
        double fx=f(x), dfx=df(x);
        if(std::abs(dfx)<1e-14) throw std::runtime_error("newton: derivative near zero");
        double dx=fx/dfx;
        x-=dx;
        if(std::abs(dx)<tol) return x;
    }
    return x;
}

// ---------- Interpolation (linear 1-D) ----------
inline double interp_linear(const std::vector<double>& xs, const std::vector<double>& ys, double x){
    if(xs.size()!=ys.size()||xs.size()<2) throw std::invalid_argument("interp_linear: need >=2 points");
    if(x<=xs.front()) return ys.front();
    if(x>=xs.back()) return ys.back();
    for(size_t i=0;i+1<xs.size();++i){
        if(x>=xs[i] && x<=xs[i+1]){
            double t=(x-xs[i])/(xs[i+1]-xs[i]);
            return ys[i]*(1-t)+ys[i+1]*t;
        }
    }
    return ys.back();
}

// ---------- Numerical differentiation (central) ----------
inline double derivative(std::function<double(double)> f, double x, double h=1e-6){
    return (f(x+h)-f(x-h))/(2*h);
}

// ---------- Numerical integration ----------
// trapezoidal and Simpson (n must be even for Simpson)
inline double integrate_trapezoidal(std::function<double(double)> f, double a, double b, int n=1000){
    if(n<=0) throw std::invalid_argument("n>0");
    double h=(b-a)/n, s=0.5*(f(a)+f(b));
    for(int i=1;i<n;++i) s+=f(a+i*h);
    return s*h;
}
inline double integrate_simpson(std::function<double(double)> f, double a, double b, int n=1000){
    if(n%2==1) ++n;
    double h=(b-a)/n;
    double s=f(a)+f(b);
    for(int i=1;i<n;++i) s+= (i%2==0?2:4)*f(a+i*h);
    return s*h/3.0;
}

} // namespace dracolix::numerics
