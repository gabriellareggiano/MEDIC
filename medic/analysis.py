import pandas as pd
from math import exp
import importlib.resources as pkg_resources
import medic.medic_model

# arguments are given in the same order as the model_params file
# matches up the coeffs 
def calc_prob(zdens1, zdens3, lddt, rama, cart):
    # collect params
    params_str = pkg_resources.read_text(medic.medic_model, "model_params.txt")
    lines = [ x.strip() for x in params_str.split('\n') if x ]
    intercept = 0
    coeff = []
    intercept = float(lines[0].split()[-1])
    coeff = [ float(line.split()[-1]) for line in lines[2:] ]
    
    # calculate prob
    fx = intercept + coeff[0]*zdens1 + coeff[1]*zdens3 + coeff[2]*lddt + coeff[3]*rama + coeff[4]*cart
    return 1/(1+exp(-1*fx))

                                     
def calculate_contributions(scores, thres):
    # these are taken from the means for over 1000 structures in EMDB
    # TODO - will i need to recalculate these for the newest version?
    lddt_mean = 0.7685407746442903
    zdens1_mean  = -0.4811404079926995
    zdens3_mean = -0.6072111874018266
    rama_mean = 0.19317735155421467
    cart_mean = 0.8337395547340672
    scores['only_lddt_probability'] = scores.apply(lamba row: calc_prob(zdens1_mean, zdens3_mean,
                                                                        row['lddt'], rama_mean, cart_mean), axis=1)
    scores['only_dens_probability'] = scores.apply(lamba row: calc_prob(row['perResZDensWin1'], row['perResZDensWin3'],
                                                                        lddt_mean, rama_mean, cart_mean), axis=1)
    scores['only_geo_probability'] = scores.apply(lamba row: calc_prob(zdens1_mean, zdens3_mean, lddt_mean,
                                                                        row['rama_prepro'], row['cart_bonded']), axis=1)
    scores['lddt_dens_probability'] = scores.apply(lamba row: calc_prob(row['perResZDensWin1'], row['perResZDensWin3'],
                                                                        row['lddt'], rama_mean, cart_mean), axis=1) 
    scores['lddt_geo_probability'] = scores.apply(lamba row: calc_prob(zdens1_mean, zdens3_mean, row['lddt'],
                                                                        row['rama_prepro'], row['cart_bonded']), axis=1)
    scores['dens_geo_probability'] = scores.apply(lamba row: calc_prob(row['perResZDensWin1'], row['perResZDensWin3'], lddt_mean,
                                                                        row['rama_prepro'], row['cart_bonded']), axis=1)

    scores['contributing_factors'] = ""
    scores.loc[scores['only_lddt_probability'] >= thres, 'contributing_factors'] = 'lddt'
    scores.loc[scores['only_dens_probability'] >= thres, 'contributing_factors'] = 'density'
    scores.loc[scores['only_geo_probability'] >= thres, 'contributing_factors'] = 'geometry'
    scores.loc[(scores['lddt_dens_probability'] >= thres) 
                    & (scores['only_lddt_probability'] < thres) 
                    & (scores['only_dens_probability'] < thres), 'contributing_factors'] = 'lddt + density'
    scores.loc[(scores['lddt_geo_probability'] >= thres) 
                    & (scores['only_lddt_probability'] < thres) 
                    & (scores['only_geo_probability'] < thres), 'contributing_factors'] = 'lddt + geometry'
    scores.loc[(scores['dens_geo_probability'] >= thres) 
                    & (scores['only_dens_probability'] < thres)  
                    & (scores['only_geo_probability'] < thres), 'contributing_factors'] = 'density + geometry'
    scores.loc[scores['contributing_factors'].str.match(""), 'contributing_factors'] = 'all scores'

    scores.drop(inplace=True, axis=1,
                    columns=['only_lddt_probability',
                            'only_dens_probability',
                            'only_geo_probability',
                            'lddt_dens_probability',
                            'lddt_geo_probability',
                            'dens_geo_probability'])
    return scores
                                     

def collect_error_info(scores, prob_coln, thres):
    scores_breakdown = calculate_contributions(scores)
    scores_breakdown['error'] = scores_breakdown[prob_coln] >= thres
    scores_breakdown['streak_id'] = (scores_breakdown['error'] != scores['error'].shift(1)).cumsum()
    errors = scores_breakdown.loc[(scores_breakdown['error'] == True)]
    grperrors = errors.groupby(['streak_id', 'chID'])
    avg = grperrors.mean()['error_probability']
    first = grperrors.first().reset_index()
    last = grperrors.last().reset_index()
                                     
    info = {"res_start": [],
                    "res_end": [],
                    "chID": [],
                    "avg_error_prob": [],
                    "contributing": []}
    #TODO - better way than with iterrows?
    for i,row in first.iterrows():
        info['res_start'].append(row['resi'])
        info['res_end'].append(last.at[i,'resi'])
        info['chID'].append(row['chID'])
        info['avg_error_prob'].append(avg.at[i,'prob'])
        info['contributing'].append(row['contributing_factors'])
     
    return info
        
                                     
def get_error_report_str(info, high_thres, low_thres):
    error_strs = []
    for st,end,ch,errp,cont in zip(info['res_start'], info['res_end'],
                                                             info['chID'], info['avg_error_prob'],
                                                             info['contributing']):
        if errp >= high_thres:
            error_type = "definite"
        elif errp < high_thres and errp >= low_thres:
            error_type = "possible"
        error_strs.append(f"{st}{ch} - {end}{ch}, {error_type} error, causes: {cont}")
    return "\n".join(error_strs)
        
