#include "Flame.H"
#include "IO/ParmParse.H"
#include "BC/Constant.H"
#include "Numeric/Stencil.H"
#include "IC/Laminate.H"
#include "IC/Constant.H"
#include "IC/PSRead.H"
#include "Numeric/Function.H"
#include "IC/Expression.H"
#include "Base/Mechanics.H"

#include <cmath>

namespace Integrator
{

    Flame::Flame() : Base::Mechanics<Model::Solid::Affine::Isotropic>() {}

    Flame::Flame(IO::ParmParse &pp) : Flame()
    {pp.queryclass(*this);}

    void 
    Flame::Parse(Flame &value, IO::ParmParse &pp)
    {
        BL_PROFILE("Integrator::Flame::Flame()");
        {
            pp.query("timestep", value.base_time);
            // These are the phase field method parameters
            // that you use to inform the phase field method.
            pp.query("pf.eps", value.pf.eps); // Burn width thickness
            pp.query("pf.kappa", value.pf.kappa); // Interface energy param
            pp.query("pf.gamma",value.pf.gamma); // Scaling factor for mobility
            pp.query("pf.lambda",value.pf.lambda); // Chemical potential multiplier
            pp.query("pf.w1", value.pf.w1); // Unburned rest energy
            pp.query("pf.w12", value.pf.w12);  // Barrier energy
            pp.query("pf.w0", value.pf.w0);    // Burned rest energy
            pp.query("pf.min_eta", value.pf.min_eta);
            
            value.bc_eta = new BC::Constant(1);
            pp.queryclass("pf.eta.bc", *static_cast<BC::Constant *>(value.bc_eta)); // See :ref:`BC::Constant`
            value.RegisterNewFab(value.eta_mf,     value.bc_eta, 1, 2, "eta", true);
            value.RegisterNewFab(value.eta_old_mf, value.bc_eta, 1, 2, "eta_old", false);
            value.RegisterNewFab(value.mdot_mf, 1, "mdot", true);

            std::string eta_bc_str = "constant";
            pp.query("pf.eta.ic.type",eta_bc_str);
            if (eta_bc_str == "constant") value.ic_eta = new IC::Constant(value.geom,pp,"pf.eta.ic.constant");
            else if (eta_bc_str == "expression") value.ic_eta = new IC::Expression(value.geom,pp,"pf.eta.ic.expression");

            std::string eta_ic_type = "constant";
            pp.query("eta.ic.type", eta_ic_type);
            if (eta_ic_type == "laminate") value.ic_eta = new IC::Laminate(value.geom, pp, "eta.ic.laminate");
            else if (eta_ic_type == "constant") value.ic_eta = new IC::Constant(value.geom, pp, "eta.ic.consntant");
	        else if (eta_ic_type == "expression") value.ic_eta = new IC::Expression(value.geom, pp, "eta.ic.expression");
	        else Util::Abort(INFO, "Invalid eta IC type", eta_ic_type);

        }
        
        {
            //IO::ParmParse pp("thermal");
            pp.query("thermal.on",value.thermal.on); // Whether to use the Thermal Transport Model
	        pp.query("elastic.on", value.elastic.on);
            pp.query("", value.thermal.bound); // System Initial Temperature
            if (true){
                pp.query("thermal.rho_ap",value.thermal.rho_ap); // AP Density
                pp.query("thermal.rho_htpb", value.thermal.rho_htpb); // HTPB Density
                pp.query("thermal.k_ap", value.thermal.k_ap); // AP Thermal Conductivity
                pp.query("thermal.k_htpb",value.thermal.k_htpb); // HTPB Thermal Conductivity
                pp.query("thermal.cp_ap", value.thermal.cp_ap); // AP Specific Heat
                pp.query("thermal.cp_htpb", value.thermal.cp_htpb); //HTPB Specific Heat

                pp.query("thermal.q0",value.thermal.q0); // Baseline heat flux

                pp.query("thermal.m_ap", value.thermal.m_ap); // AP Pre-exponential factor for Arrhenius Law
                pp.query("thermal.m_htpb", value.thermal.m_htpb); // HTPB Pre-exponential factor for Arrhenius Law
	            pp.query("thermal.m_comb", value.thermal.m_comb);
                pp.query("thermal.E_ap", value.thermal.E_ap); // AP Activation Energy for Arrhenius Law
                pp.query("thermal.E_htpb", value.thermal.E_htpb); // HTPB Activation Energy for Arrhenius Law

                pp.query("thermal.hc", value.thermal.hc); // Used to change heat flux units
	            pp.query("thermal.massfraction", value.thermal.massfraction); // Systen AP mass fraction
	            pp.query("thermal.mlocal_ap", value.thermal.mlocal_ap);
	            pp.query("thermal.mlocal_htpb", value.thermal.mlocal_htpb);
	            pp.query("thermal.mlocal_comb", value.thermal.mlocal_comb);

	            pp.query("thermal.T_fluid", value.thermal.T_fluid); // Temperature of the Standin Fluid 

	            pp.query("thermal.modeling_ap", value.thermal.modeling_ap);
	            pp.query("thermal.modeling_htpb", value.thermal.modeling_htpb);
	    
                value.bc_temp = new BC::Constant(1);
                pp.queryclass("thermal.temp.bc", *static_cast<BC::Constant *>(value.bc_temp));
                value.RegisterNewFab(value.temp_mf,     value.bc_temp, 1, 2, "temp", true);
                value.RegisterNewFab(value.temp_old_mf, value.bc_temp, 1, 2, "temp_old", false);
                value.RegisterNewFab(value.temps_mf, value.bc_temp, 1, 2, "temps", true);
	            value.RegisterNewFab(value.temps_old_mf, value.bc_temp, 1, 2, "temps_old", false);
	    
                value.RegisterNewFab(value.mob_mf, 1, "mob", true);
                value.RegisterNewFab(value.alpha_mf, 1,"alpha",true);  
                value.RegisterNewFab(value.heatflux_mf, 1, "heatflux", true);

	            value.RegisterNewFab(value.laser_mf, 1, "laser", true);
  	            std::string laser_ic_type = "constant";
	            pp.query("laser.ic.type", laser_ic_type);
	            if (laser_ic_type == "expression") value.ic_laser = new IC::Expression(value.geom,pp,"laser.ic.expression");
	            else if (laser_ic_type == "constant") value.ic_laser = new IC::Constant(value.geom, pp, "laser.ic.constant");
	            else Util::Abort(INFO, "Invalid eta IC type", laser_ic_type);
            }
        }

        {
            pp.query("pressure.P", value.pressure.P);
            //if (thermal.on == true)
            {
                pp.query("pressure.a1"    , value.pressure.arrhenius.a1);
                pp.query("pressure.a2"    , value.pressure.arrhenius.a2);
                pp.query("pressure.a3"    , value.pressure.arrhenius.a3);
                pp.query("pressure.b1"    , value.pressure.arrhenius.b1);
                pp.query("pressure.b2"    , value.pressure.arrhenius.b2);
                pp.query("pressure.b3"    , value.pressure.arrhenius.b3);
                pp.query("pressure.c1"    , value.pressure.arrhenius.c1);
	            pp.query("pressure.mob_ap", value.pressure.arrhenius.mob_ap);
                pp.query("pressure.dependency", value.pressure.arrhenius.dependency);
            }
            //else if (thermal.on == false)
            {
                pp.query("pressure.r_ap"  , value.pressure.power.r_ap  );
                pp.query("pressure.r_htpb", value.pressure.power.r_htpb);
                pp.query("pressure.r_comb", value.pressure.power.r_comb);
                pp.query("pressure.n_ap"  , value.pressure.power.n_ap  );
                pp.query("pressure.n_htpb", value.pressure.power.n_htpb);
                pp.query("pressure.n_comb", value.pressure.power.n_comb);
            }
        }

        // Refinement criterion for eta field
        pp.query("amr.refinement_criterion", value.m_refinement_criterion); 
        // Refinement criterion for temperature field
        pp.query("amr.refinement_criterion_temp", value.t_refinement_criterion); 
        // Eta value to restrict the refinament for the temperature field
        pp.query("amr.refinament_restriction", value.t_refinement_restriction);
        // small value
        pp.query("small", value.small);
               
        {
            // The material field is referred to as :math:`\phi(\mathbf{x})` and is 
            // specified using these parameters. 
            //IO::ParmParse pp("phi.ic");
            std::string type = "packedspheres";
            pp.query("phi.ic.type", type); // IC type (psread, laminate, constant)
            if (type == "psread"){
                value.ic_phi = new IC::PSRead(value.geom, pp, "phi.ic.psread");
                pp.query("phi.ic.psread.eps", value.zeta);
                pp.query("phi.zeta_0", value.zeta_0);
                }
            else if (type == "laminate"){
                value.ic_phi = new IC::Laminate(value.geom,pp,"phi.ic.laminate");
                pp.query("phi.ic.laminate.eps", value.zeta);
                pp.query("phi.zeta_0", value.zeta_0);
                }
	        else if (type == "expression"){
	            value.ic_phi = new IC::Expression(value.geom,pp,"phi.ic.expression");
		        pp.query("phi.zeta_0", value.zeta_0);
		        pp.query("phi.zeta", value.zeta);
	        }
	        //else if (type == "cuboid"){
	        //    value.ic_phi = new IC::Cuboid(value.geom, pp, "phi.ic.cuboid");
	        //    }
	        //else if (type == "sphere"){
	        //    value.ic_phi = new IC::Sphere(value.geom, pp, "phi.ic.sphere");
	        //    } 	    
            else if (type == "constant")  value.ic_phi = new IC::Constant(value.geom,pp,"phi.ic.constant");
            else Util::Abort(INFO,"Invalid IC type ",type);           
            value.RegisterNewFab(value.phi_mf, value.bc_eta, 1, 2, "phi", true);
        }

        pp.queryclass<Base::Mechanics<Model::Solid::Affine::Isotropic>>("elastic",value);
        if (value.m_type  != Type::Disable)
        {
            pp.queryclass("model_ap",value.elastic.model_ap);
            pp.queryclass("model_htpb",value.elastic.model_htpb);

	        value.bc_psi = new BC::Nothing();
	        value.RegisterNewFab(value.psi_mf,value.bc_psi,1,2,"psi",true);
	        value.psi_on = true;
        }

        }

    void Flame::Initialize(int lev)
    {
        BL_PROFILE("Integrator::Flame::Initialize");
        Util::Message(INFO,m_type);
	    Base::Mechanics<Model::Solid::Affine::Isotropic>::Initialize(lev);

        temp_mf[lev]->setVal(thermal.bound);
        temp_old_mf[lev]->setVal(thermal.bound);
	    temps_mf[lev] ->setVal(thermal.bound);
	    temps_old_mf[lev] -> setVal(thermal.bound);

        alpha_mf[lev]->setVal(0.0);
        mob_mf[lev] -> setVal(0.0);

        ic_eta->Initialize(lev,eta_mf);
        ic_eta->Initialize(lev,eta_old_mf);

        mdot_mf[lev]->setVal(0.0);

        heatflux_mf[lev] -> setVal(0.0);

        ic_phi->Initialize(lev, phi_mf);
	    thermal.T_fluid = thermal.bound;

	    thermal.w1 = 0.2 * pressure.P + 0.9;

	    ic_laser->Initialize(lev,laser_mf);
	
	    if (elastic.on) psi_mf[lev]->setVal(1.0);

	    thermal.mlocal_htpb = 685000.0 - 850e3*thermal.massfraction;
    }
    
    void Flame::UpdateModel(int /*a_step*/)
    {
        if (m_type == Base::Mechanics<Model::Solid::Affine::Isotropic>::Type::Disable) return;


        for (int lev = 0; lev <= finest_level; ++lev)
        {
	        //psi_mf[lev]->setVal(1.0);
            phi_mf[lev]->FillBoundary();
            eta_mf[lev]->FillBoundary();
            temp_mf[lev]->FillBoundary();
            
            for (MFIter mfi(*model_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box bx = mfi.nodaltilebox();
		        bx.grow(1);
                amrex::Array4<model_type>        const &model = model_mf[lev]->array(mfi);
                amrex::Array4<const Set::Scalar> const &phi = phi_mf[lev]->array(mfi);
 
                if (elastic.on)
                {
                    amrex::Array4<const Set::Scalar> const &temp = temp_mf[lev]->array(mfi);
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        Set::Scalar phi_avg = Numeric::Interpolate::CellToNodeAverage(phi,i,j,k,0);
                        Set::Scalar temp_avg = Numeric::Interpolate::CellToNodeAverage(temp,i,j,k,0);
                        model_type model_ap = elastic.model_ap;
                        model_ap.F0 *= (temp_avg-thermal.bound);

                        model_type model_htpb = elastic.model_htpb;
                        model_htpb.F0 *= (temp_avg-thermal.bound);
                        model(i,j,k) = model_ap*phi_avg + model_htpb*(1.-phi_avg);
                    });
                }
                else
                {
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        Set::Scalar phi_avg = Numeric::Interpolate::CellToNodeAverage(phi,i,j,k,0);
                        model_type model_ap = elastic.model_ap;
                        model_ap.F0 *= Set::Matrix::Zero();
                        model_type model_htpb = elastic.model_htpb;
                        model_htpb.F0 *= Set::Matrix::Zero();
                        model(i,j,k) = model_ap*phi_avg + model_htpb*(1.-phi_avg);
                    });
                }
            }

            Util::RealFillBoundary(*model_mf[lev], geom[lev]);
            
            psi_mf[lev]->setVal(1.0);
            amrex::MultiFab::Copy(*psi_mf[lev],*eta_mf[lev],0,0,1,psi_mf[lev]->nGrow());
        }
    }
    
    void Flame::TimeStepBegin(Set::Scalar a_time, int a_iter)
    {
        BL_PROFILE("Integrator::Flame::TimeStepBegin");
        Base::Mechanics<Model::Solid::Affine::Isotropic>::TimeStepBegin(a_time,a_iter);
        for (int lev = 0; lev <= finest_level; ++lev)
	    ic_laser->Initialize(lev,laser_mf,a_time);
    }


    void Flame::Advance(int lev, Set::Scalar time, Set::Scalar dt)
    {
        BL_PROFILE("Integrador::Flame::Advance");
	    Base::Mechanics<Model::Solid::Affine::Isotropic>::Advance(lev,time,dt);
        const Set::Scalar *DX = geom[lev].CellSize();

        if (true) // (lev == finest_level) //(true)
        {
            if (thermal.on) 
            {
                std::swap(eta_old_mf[lev], eta_mf[lev]);
                std::swap(temp_old_mf[lev], temp_mf[lev]);
	            std::swap(temps_old_mf[lev], temps_mf[lev]);

                Numeric::Function::Polynomial<4> w(pf.w0, 
                                            0.0, 
                                            -5.0 * pf.w1 + 16.0 * pf.w12 - 11.0 * pf.w0,
                                            14.0 * pf.w1 - 32.0 * pf.w12 + 18.0 * pf.w0,
                                            -8.0 * pf.w1 + 16.0 * pf.w12 -  8.0 * pf.w0 );
                Numeric::Function::Polynomial<3> dw = w.D();

                for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
                {               
                    const amrex::Box &bx = mfi.tilebox();
                    // Phase fields
                    amrex::Array4<Set::Scalar> const &etanew = (*eta_mf[lev]).array(mfi);
                    amrex::Array4<const Set::Scalar> const &eta = (*eta_old_mf[lev]).array(mfi);
                    amrex::Array4<const Set::Scalar> const &phi = (*phi_mf[lev]).array(mfi);
                    // Heat transfer fields
                    amrex::Array4<Set::Scalar>       const &tempnew = (*temp_mf[lev]).array(mfi);
                    amrex::Array4<const Set::Scalar> const &temp    = (*temp_old_mf[lev]).array(mfi);
                    amrex::Array4<Set::Scalar>       const &alpha   = (*alpha_mf[lev]).array(mfi);
		            amrex::Array4<Set::Scalar>       const &tempsnew= (*temps_mf[lev]).array(mfi);
		            amrex::Array4<Set::Scalar>       const &temps   = (*temps_old_mf[lev]).array(mfi);
		            amrex::Array4<Set::Scalar>       const &laser   = (*laser_mf[lev]).array(mfi);
                    // Diagnostic fields
                    amrex::Array4<Set::Scalar> const  &mob = (*mob_mf[lev]).array(mfi);
                    amrex::Array4<Set::Scalar> const &mdot = (*mdot_mf[lev]).array(mfi);
                    amrex::Array4<Set::Scalar> const &heatflux = (*heatflux_mf[lev]).array(mfi);
		            // Constants
		            Set::Scalar zeta_2 = 0.000045 - pressure.P * 6.42e-6;
		            Set::Scalar zeta_1; 
		            if (pressure.arrhenius.dependency == 1) zeta_1 = zeta_2;
		            else zeta_1 = zeta_0; 
		            Set::Scalar k1 = pressure.arrhenius.a1 * pressure.P + pressure.arrhenius.b1 - zeta_1 / zeta;
		            Set::Scalar k2 = pressure.arrhenius.a2 * pressure.P + pressure.arrhenius.b2 - zeta_1 / zeta;
		            Set::Scalar k3 = 4.0 * log((pressure.arrhenius.c1 * pressure.P * pressure.P + pressure.arrhenius.a3 * pressure.P + pressure.arrhenius.b3) - k1 / 2.0 - k2 / 2.0);
                		
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        Set::Scalar eta_lap = Numeric::Laplacian(eta, i, j, k, 0, DX);  
                        Set::Scalar K   = thermal.modeling_ap * thermal.k_ap * phi(i,j,k) + thermal.modeling_htpb * thermal.k_htpb * (1.0 - phi(i,j,k)); // Calculate effective thermal conductivity
                        Set::Scalar rho = thermal.rho_ap * phi(i,j,k) + thermal.rho_htpb * (1.0 - phi(i,j,k)); // No special interface mixure rule is needed here.
                        Set::Scalar cp  = thermal.cp_ap  * phi(i,j,k) + thermal.cp_htpb  * (1.0 - phi(i,j,k));            
		                Set::Scalar df_deta = ( (pf.lambda/pf.eps) * dw( eta(i,j,k) ) - pf.eps * pf.kappa * eta_lap );
		                etanew(i, j, k) = eta(i, j, k) - mob(i,j,k) * dt * df_deta;
                        alpha(i,j,k) =  K / rho / cp; // Calculate thermal diffusivity and store in fiel
		                mdot(i,j,k) = rho * fabs(eta(i,j,k) - etanew(i,j,k)) / dt; // deta/dt
		            });
		
		            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
		            {
		                Set::Scalar K = thermal.modeling_ap * thermal.k_ap * phi(i,j,k) + thermal.modeling_htpb * thermal.k_htpb * (1.0 - phi(i,j,k));
		                Set::Scalar qflux = k1 * phi(i,j,k) +
		                        k2 * (1.0 - phi(i,j,k)) +
		                        (zeta_1 / zeta) * exp(k3 * phi(i,j,k) * (1.0 - phi(i,j,k)));
		                Set::Scalar mlocal = (thermal.mlocal_ap) * phi(i,j,k) + (thermal.mlocal_htpb) * (1.0 - phi(i,j,k)) + thermal.mlocal_comb * phi(i,j,k) * (1.0 - phi(i,j,k));
		                Set::Scalar mdota = fabs(mdot(i,j,k));
		                Set::Scalar mbase = tanh(4.0*mdota / mlocal);		     
		                heatflux(i,j,k) = (thermal.hc * mbase * qflux + laser(i,j,k) ) / K;
		            });
                     
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
		                auto sten = Numeric::GetStencil(i,j,k,bx);
                        Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
                        Set::Vector grad_temp = Numeric::Gradient(temp, i, j, k, 0, DX);
                        Set::Scalar lap_temp = Numeric::Laplacian(temp, i, j, k, 0, DX);
                        Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
                        Set::Vector grad_alpha = Numeric::Gradient(alpha,i,j,k,0,DX, sten);                		    
                        Set::Scalar dTdt = 0.0;
		                dTdt += grad_eta.dot(grad_temp * alpha(i,j,k));
		                dTdt += grad_alpha.dot(eta(i,j,k) * grad_temp);
		                dTdt += eta(i,j,k) * alpha(i,j,k) * lap_temp;
		                dTdt += alpha(i,j,k) * heatflux(i,j,k) * grad_eta_mag;
		                Set::Scalar Tsolid;
		                Tsolid = dTdt + temps(i,j,k) * (etanew(i,j,k) - eta(i,j,k)) / dt; 
		                tempsnew(i,j,k) = temps(i,j,k) + dt * Tsolid;
		                tempnew(i,j,k) = etanew(i,j,k) * tempsnew(i,j,k) + (1.0 - etanew(i,j,k)) * thermal.T_fluid;

                    });
            
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
		                Set::Scalar L;		    
                        if (pressure.arrhenius.mob_ap == 1) L = thermal.m_ap * pressure.P * exp(-thermal.E_ap / tempnew(i,j,k) ) * phi(i,j,k);
		                else L = thermal.m_ap * exp(-thermal.E_ap / tempnew(i,j,k) ) * phi(i,j,k);
		                L += thermal.m_htpb * exp(-thermal.E_htpb / tempnew(i,j,k) ) * (1.0 - phi(i,j,k));
		                L += thermal.m_comb * (0.5 * tempnew(i,j,k) / thermal.bound) * phi(i,j,k) * (1.0 - phi(i,j,k) ); 		    
		                if(tempnew(i,j,k) <= thermal.bound) mob(i,j,k) = 0;
		                else mob(i,j,k) = L;
                    });
                } // MFi For loop 
            } // thermal IF
            else
            {
                std::swap(eta_old_mf[lev], eta_mf[lev]);
                Numeric::Function::Polynomial<4> w(pf.w0,
                    0.0,
                    -5.0 * pf.w1 + 16.0 * pf.w12 - 11.0 * pf.w0,
                    14.0 * pf.w1 - 32.0 * pf.w12 + 18.0 * pf.w0,
                    -8.0 * pf.w1 + 16.0 * pf.w12 - 8.0  * pf.w0
                );
                Numeric::Function::Polynomial<3> dw = w.D();

                for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
                {
                    const amrex::Box& bx = mfi.tilebox();
                    //Phase Fields
                    amrex::Array4<Set::Scalar> const& etanew = (*eta_mf[lev]).array(mfi);
                    amrex::Array4<Set::Scalar> const& eta   = (*eta_old_mf[lev]).array(mfi); 
                    amrex::Array4<Set::Scalar> const& phi   = (*phi_mf[lev]).array(mfi);

                    Set::Scalar fmod_ap   = pressure.power.r_ap   * pow(pressure.P, pressure.power.n_ap);
                    Set::Scalar fmod_htpb = pressure.power.r_htpb * pow(pressure.P, pressure.power.n_htpb);
                    Set::Scalar fmod_comb = pressure.power.r_comb * pow(pressure.P, pressure.power.n_comb);

                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        Set::Scalar fs_actual;
                        fs_actual = fmod_ap * phi(i,j,k) 
                                  + fmod_htpb * (1.0 - phi(i,j,k))
                                  + 4.0 * fmod_comb * phi(i,j,k) * (1.0 - phi(i,j,k));
                        Set::Scalar L = fs_actual / pf.gamma / (pf.w1 - pf.w0);
                        Set::Scalar eta_lap = Numeric::Laplacian(eta, i, j, k, 0, DX);
                        Set::Scalar df_deta = ((pf.lambda / pf.eps) * dw(eta(i,j,k)) - pf.eps * pf.kappa * eta_lap);
                        etanew(i,j,k) = eta(i,j,k) - L * dt * df_deta;

                        if (etanew(i,j,k) > eta(i,j,k)) etanew(i,j,k) = eta(i,j,k);
                    });
                } // MFI For Loop 
            } // thermal ELSE      
        }// First IF
    } //Function


    void Flame::TagCellsForRefinement(int lev, amrex::TagBoxArray &a_tags, Set::Scalar time, int ngrow)
    {
        BL_PROFILE("Integrator::Flame::TagCellsForRefinement");
	    Base::Mechanics<Model::Solid::Affine::Isotropic>::TagCellsForRefinement(lev,a_tags,time,ngrow);
        
        const Set::Scalar *DX = geom[lev].CellSize();
        Set::Scalar dr = sqrt(AMREX_D_TERM(DX[0] * DX[0], +DX[1] * DX[1], +DX[2] * DX[2]));

        // Eta criterion for refinement
        for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();
            amrex::Array4<char> const &tags = a_tags.array(mfi);
            amrex::Array4<const Set::Scalar> const &eta = (*eta_mf[lev]).array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                Set::Vector gradeta = Numeric::Gradient(eta, i, j, k, 0, DX);
                if (gradeta.lpNorm<2>() * dr * 2 > m_refinement_criterion)
                    tags(i, j, k) = amrex::TagBox::SET; });
        }

        // Thermal criterion for refinement          
        for (amrex::MFIter mfi(*temp_mf[lev], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();
            amrex::Array4<char> const &tags = a_tags.array(mfi);
            amrex::Array4<const Set::Scalar> const &temp = (*temp_mf[lev]).array(mfi);
            amrex::Array4<const Set::Scalar> const &eta  = (*eta_mf[lev]).array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                Set::Vector tempgrad = Numeric::Gradient(temp, i, j, k, 0, DX);
                if (tempgrad.lpNorm<2>() * dr  > t_refinement_criterion && eta(i,j,k) >= t_refinement_restriction)
                                tags(i, j, k) = amrex::TagBox::SET;
            });
        }         
    }

    void Flame::Regrid(int lev, Set::Scalar time)
    {
        BL_PROFILE("Integrator::Flame::Regrid");
        //if (lev < finest_level) return;
        //phi_mf[lev]->setVal(0.0);
        ic_phi->Initialize(lev, phi_mf,time);
    } 

    void Flame::Integrate(int amrlev, Set::Scalar /*time*/, int /*step*/,
                                            const amrex::MFIter &mfi, const amrex::Box &box)
    {
        BL_PROFILE("Flame::Integrate");
        const Set::Scalar *DX = geom[amrlev].CellSize();
        Set::Scalar dv = AMREX_D_TERM(DX[0], *DX[1], *DX[2]);
        amrex::Array4<amrex::Real> const &eta = (*eta_mf[amrlev]).array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) 
        {
            volume += eta(i, j, k, 0) * dv;
            Set::Vector grad = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar normgrad = grad.lpNorm<2>();
            Set::Scalar da = normgrad * dv;
            area += da;
        });
    }
} // namespace Integrator


