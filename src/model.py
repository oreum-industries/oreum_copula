# src.model.py
# copyright 2021 Oreum OÜ
import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
from oreum_core import model

class ModelA(model.BasePYMC3Model):
    """ Basic model to demonstrate essential architecture (priors, marginal 
        likelihoods, copula). As used by and described in 10_Model_A.ipynb.
                
        Does not faciliate: linear sub-models on the parameters of marginals 
        (this is intercept only); zero-inflated marginals; missing data.
                
        Uses core components from oreum_core.model.BasePYMC3Model
        
        NOTE: 
            1. shapes are different throughout when only using intercept:
                (n, j) becomes (n, ) when j=1
            2. arviz doesn't like dims of len 1 (for intercept) so still create 
                the same dim naming, but can only use in PPC reconstruction
    """
    version = '0.1.0'
    name = 'mdla'

    def __init__(self, obs_m1: pd.DataFrame, 
                       obs_m2: pd.DataFrame, *args, **kwargs):
        """ Expects 2 dataframes for observations, 1 per linear submodel, 
            each arranged as: pd.DataFrame(mx_en, mx_exs) """
        super().__init__(*args, **kwargs)
        
        self.sample_kws['init'] = 'adapt_diag' # sensitive startpos avoid jitter
       
        self.obs_m1 = obs_m1
        self.cords_m1 = {'names_j_m1': np.array(['intercept']),
                            'obs_id': np.arange(len(self.obs_m1)),
                            'c_m1m2': ['c_m1', 'c_m2']}

        self.obs_m2 = obs_m2
        self.cords_m2 = {'names_j_m2': np.array(['intercept'])}


    def _build(self):
        """ Builds and returns the model. Also sets self.model """
       
        with pm.Model(coords={**self.cords_m1, **self.cords_m2}) as self.model:
        
            ### Create Data objects (observations)            
            y_m1 = pm.Data('y_m1', self.obs_m1['m1'].values)
            x_m1 = pm.Data('x_m1', self.obs_m1['intercept'].values)
            
            y_m2 = pm.Data('y_m2', self.obs_m2['m2'].values)
            x_m2 = pm.Data('x_m2', self.obs_m2['intercept'].values)

            ### Create marginals, parameterised to obs (intercept-only)
            m1_b = pm.Normal('m1_b', mu=-1., sigma=1.)
            m1_mu = tt.dot(m1_b, x_m1.T)
            m1_sigma = pm.InverseGamma('m1_sigma', alpha=11., beta=10.)
            m1_dist = model.Lognormal.dist(mu=m1_mu, sigma=m1_sigma)

            m2_b = pm.Normal('m2_b', mu=2., sigma=1.)  # keep narrow-ish
            m2_mu = tt.dot(m2_b, x_m2.T)
            m2_sigma = pm.InverseGamma('m2_sigma', alpha=11., beta=20.) # keep narrow ~ [1, 3]
            m2_dist = model.Lognormal.dist(mu=m2_mu, sigma=m2_sigma)
       
            ### Transform obs marginals to uniform through marginal CDFs
            y_m1u = pm.Deterministic('y_m1u', m1_dist.cdf(y_m1), dims='obs_id')
            y_m2u = pm.Deterministic('y_m2u', m2_dist.cdf(y_m2), dims='obs_id')

            ### Transform uniformed marginals to MvN to fit against
            ### the likelihood of a random-variates copula.
            # Note for prior predictive, note the use of 
            # model.distributions.CLIP_U_AWAY_FROM_ZERO_ONE_FOR_INVCDFS 
            # to prevent infs in copula_obs. This is acute in the tails of the 
            # marginals: uniform transform hits 1 and can round slightly over 1
            norm_dist = model.Normal.dist(mu=0., sigma=1.)
            y_cop = pm.Deterministic('y_cop', 
                        norm_dist.invcdf(tt.stack([y_m1u, y_m2u]).T),
                        dims=('obs_id', 'c_fs'))

            ### Evidence the independent marginals via jacobian of f_inv
            # Note this separate from the copula likelihood
            m1_jcd = pm.Potential('m1_jcd', model.log_jcd(y_cop[:, 0], y_m1))
            m2_jcd = pm.Potential('m2_jcd', model.log_jcd(y_cop[:, 1], y_m2))

            ### Set cholesky cov standard dev prior
            sd_dist = pm.InverseGamma.dist(alpha=11., beta=10.)
            
            ### Create cholesky cov (use convenience method to unpack chol)
            chol, corr_, stds_ = pm.LKJCholeskyCov('lkjcc', n=2, eta=2.,
                                            sd_dist=sd_dist, compute_corr=True)
                
            ### Create copula likelihood (use centered parameterisation)
            cop_dist = pm.MvNormal.dist(mu=tt.zeros(2), chol=chol, 
                                           shape=(len(self.obs_freq), 2))
            cop_like = pm.Potential('cop_like', cop_dist.logp(y_cop))
                
        self.rvs_for_posterior_plots = [
            'm1_b', 'm1_sigma', 
            'm1_b', 'm2_sigma', 
            'lkjcc', 'lkjcc_corr', 'lkjcc_stds', 
        ]
        
        return self.model

