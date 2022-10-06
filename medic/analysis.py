from math import exp
import importlib.resources as pkg_resources
import pandas as pd

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
    scores['only_lddt_probability'] = scores.apply(lambda row: calc_prob(zdens1_mean, zdens3_mean,
                                                                        row['lddt'], rama_mean, cart_mean), axis=1)
    scores['only_dens_probability'] = scores.apply(lambda row: calc_prob(row['perResZDensWin1'], row['perResZDensWin3'],
                                                                        lddt_mean, rama_mean, cart_mean), axis=1)
    scores['only_geo_probability'] = scores.apply(lambda row: calc_prob(zdens1_mean, zdens3_mean, lddt_mean,
                                                                        row['rama_prepro'], row['cart_bonded']), axis=1)
    scores['lddt_dens_probability'] = scores.apply(lambda row: calc_prob(row['perResZDensWin1'], row['perResZDensWin3'],
                                                                        row['lddt'], rama_mean, cart_mean), axis=1) 
    scores['lddt_geo_probability'] = scores.apply(lambda row: calc_prob(zdens1_mean, zdens3_mean, row['lddt'],
                                                                        row['rama_prepro'], row['cart_bonded']), axis=1)
    scores['dens_geo_probability'] = scores.apply(lambda row: calc_prob(row['perResZDensWin1'], row['perResZDensWin3'], lddt_mean,
                                                                        row['rama_prepro'], row['cart_bonded']), axis=1)

    scores['contributing_factors'] = "none"
    scores.loc[(scores['only_lddt_probability'] >= thres)
                & (scores['contributing_factors'].str.match("none")), 'contributing_factors'] = 'lddt'
    scores.loc[(scores['only_dens_probability'] >= thres)
                & (scores['contributing_factors'].str.match("none")), 'contributing_factors'] = 'density'
    scores.loc[(scores['only_geo_probability'] >= thres)
                & (scores['contributing_factors'].str.match("none")), 'contributing_factors'] = 'geometry'
    scores.loc[(scores['lddt_dens_probability'] >= thres) 
                    & (scores['only_lddt_probability'] < thres) 
                    & (scores['only_dens_probability'] < thres)
                    & (scores['contributing_factors'].str.match("none")), 'contributing_factors'] = 'lddt + density'
    scores.loc[(scores['lddt_geo_probability'] >= thres) 
                    & (scores['only_lddt_probability'] < thres) 
                    & (scores['only_geo_probability'] < thres)
                    & (scores['contributing_factors'].str.match("none")), 'contributing_factors'] = 'lddt + geometry'
    scores.loc[(scores['dens_geo_probability'] >= thres) 
                    & (scores['only_dens_probability'] < thres)  
                    & (scores['only_geo_probability'] < thres)
                    & (scores['contributing_factors'].str.match("none")), 'contributing_factors'] = 'density + geometry'
    scores.loc[scores['contributing_factors'].str.match("none"), 'contributing_factors'] = 'lddt + density + geometry'
    scores.to_csv("test_analysis.csv")
    scores.drop(inplace=True, axis=1,
                    columns=['only_lddt_probability',
                            'only_dens_probability',
                            'only_geo_probability',
                            'lddt_dens_probability',
                            'lddt_geo_probability',
                            'dens_geo_probability'])
    return scores


def get_group_contributors(df):
    vals = list(df['contributing_factors'].unique())
    final = set()
    for v in vals:
        subs = [ x.strip() for x in v.split('+') if x.strip() not in final]
        for s in subs:
            final.add(s)
    final = sorted(list(final))
    return ' + '.join(final)


def collect_error_info(scores, prob_coln, thres):
    scores_breakdown = calculate_contributions(scores, thres)
    # group by error segment and chain ID (chain id will break up mistakes across chains)
    scores_breakdown['error'] = scores_breakdown[prob_coln] >= thres
    scores_breakdown['streak_id'] = (scores_breakdown['error'] != scores_breakdown['error'].shift(1)).cumsum()
    errors = scores_breakdown.loc[(scores_breakdown['error'] == True)]
    grperrors = errors.groupby(['streak_id', 'chID'])

    # collect important information about segments
    segment_info = pd.DataFrame()
    segment_info['res_start'] = grperrors.first().reset_index()['resi']
    segment_info['res_end'] = grperrors.last().reset_index()['resi']
    segment_info['chID'] = grperrors.first().reset_index()['chID']
    segment_info['avg_error_prob'] = grperrors.mean()[prob_coln].reset_index()[prob_coln]
    segment_info['contribution'] = grperrors.apply(get_group_contributors).reset_index()[0]
    return segment_info


def get_seg_str(start, end, chain, prob, contr, high_thres, low_thres):
    err_type = ""
    if prob >= high_thres:
        err_type = "definite"
    elif low_thres <= prob < high_thres:
        err_type = "possible"
    seg_str = ""
    if start == end:
        seg_str = f"{start}{chain}"
    else:
        seg_str = f"{start}{chain} - {end}{chain}"
    return f"{seg_str}, {err_type} error\n\tcauses: {contr}"

                                     
def get_error_report_str(info, high_thres, low_thres):
    error_strs = '\n\n'.join(
                    info.apply(
                        lambda row: get_seg_str(row['res_start'], row['res_end'], 
                                                row['chID'], row['avg_error_prob'],
                                                row['contribution'],
                                                high_thres, low_thres), axis=1))
    return error_strs
        

# for debugging
def commandline_main():
    import sys

    data = pd.read_csv(sys.argv[1])
    info = collect_error_info(data, "error_probability", 0.60)
    summary = get_error_report_str(info, 0.78, 0.60)
    print('------------------------- ERROR SUMMARY -------------------------')
    print(summary)
    print('-----------------------------------------------------------------')


if __name__ == "__main__":
    commandline_main()