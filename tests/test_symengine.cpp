// -*- coding: utf-8; compile-command: "g++ -std=c++11 -I../pycvodes/include -I../external/anyode/include -I/opt/symengine-git/include test_symengine.cpp /opt/symengine-git/lib/libsymengine.a -lLLVM -lgmp -lsundials_cvodes -lsundials_nvecserial -lopenblas; ./a.out" -*-
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include "catch.hpp"
#include "cvodes_anyode.hpp"
#include "testing_utils.hpp"

#include <symengine/llvm_double.h>

namespace se = SymEngine;
using se_basic = se::RCP<const se::Basic>;

struct SymbolicSys : public AnyODE::OdeSysBase<double>  {
    int m_ny, m_nparams;
    se::vec_basic m_p, m_args;
    se::DenseMatrix m_E, m_J;
    se::LLVMDoubleVisitor m_rhs_vis;
    se::LLVMDoubleVisitor m_jac_vis;
    std::vector<double> m_buffer;
    SymbolicSys(se_basic t, se::vec_basic y, se::vec_basic p, se::vec_basic exprs) :
        m_buffer(1 + y.size() + p.size()), m_ny(y.size()), m_nparams(p.size())
    {
        if (y.size() != exprs.size()) throw std::logic_error("incompatible sizes");
        se::DenseMatrix E(exprs.size(), 1, exprs);
        se::DenseMatrix Y(y.size(), 1, y);
        se::DenseMatrix Jrow, Jcol;
        jacobian(E, Y, Jrow);
        transpose_dense(Jrow, Jcol);
        m_args.push_back(t);
        m_args.insert(m_args.end(), y.begin(), y.end());
        m_args.insert(m_args.end(), p.begin(), p.end());
        m_rhs_vis.init(m_args, exprs);
        m_jac_vis.init(m_args, Jcol.as_vec_basic());
    }
    void set_param_values(const double * const p){
        for (int i=0; i<m_nparams; ++i)
            m_buffer[i+get_ny()+1] = p[i];
    }
    int get_ny() const override { return m_ny; }
    AnyODE::Status rhs(double t, const double * const y, double * const f) override {
        m_buffer[0] = t;
        for (int i=0; i<get_ny(); ++i)
            m_buffer[i+1] = y[i];
        m_rhs_vis.call(f, &m_buffer[0]);
        return AnyODE::Status::success;
    }
    AnyODE::Status dense_jac_cmaj(double t,
                                  const double * const y,
                                  const double * const fy,
                                  double * const jac,
                                  long int ldim,
                                  double * const dfdt=nullptr) override {
        m_buffer[0] = t;
        for (int i=0; i<get_ny(); ++i)
            m_buffer[i+1] = y[i];
        m_jac_vis.call(jac, &m_buffer[0]);
        return AnyODE::Status::success;
    }
};

TEST_CASE( "decay_adaptive", "[simple_adaptive]" ) {
    auto t = se::symbol("t");
    auto y0 = se::symbol("y0");
    auto y1 = se::symbol("y1");
    auto y2 = se::symbol("y2");
    auto p0 = se::symbol("p0");
    auto p1 = se::symbol("p1");
    auto r0 = se::mul(p0, y0);
    auto r1 = se::mul(p1, y1);
    auto dy0dt = se::mul(se::integer(-1), r0);
    auto dy1dt = se::add(r0, se::mul(se::integer(-1), r1));
    auto dy2dt = r1;
    se::vec_basic exprs = {dy0dt, dy1dt, dy2dt};
    SymbolicSys odesys(t, {y0, y1, y2}, {p0, p1}, exprs);
    std::vector<int> root_indices;
    int td = 1;
    double * xyout = (double *)malloc(td*(odesys.get_ny()+1)*sizeof(double));
#define xout(ti) xyout[4*ti]
#define yout(ti, i) xyout[4*ti + 1 + i]
    xout(0) = 0.0;
    yout(0, 0) = 1.0;
    yout(0, 1) = 0.0;
    yout(0, 2) = 0.0;
    std::vector<double> pvals = {.42, .17};
    odesys.set_param_values(pvals.data());
    auto nout = cvodes_anyode::simple_adaptive(&xyout, &td, &odesys, {1e-12}, 1e-12, cvodes_cxx::LMM::BDF, 1.0, root_indices);
    for (int i = 0; i < nout; ++i){
        REQUIRE( std::abs(std::exp(-pvals[0]*xout(i)) - yout(i, 0)) < 1e-8 );
    }
    REQUIRE( odesys.last_integration_info["n_steps"] > 1 );
    REQUIRE( odesys.last_integration_info["n_steps"] >= nout );
    REQUIRE( odesys.last_integration_info["n_steps"] < 997 );
#undef xout
#undef yout
    free(xyout);
    for (auto &it : odesys.last_integration_info)
        std::cout << it.first << ": " << it.second << "\n";
}
