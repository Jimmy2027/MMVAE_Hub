import sys
import os



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


plt.style.use('seaborn-colorblind');
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

models = ['poe', 'moe', 'joint']

DIR_BASE = '/home/thomas/polybox/PhD/projects/multimodality/results_jointelbo';

def create_ellipse_mean_std(ax, mean_coh, mean_lh, std_coh, std_lh, color):
    num_points = len(mean_coh);
    for k in range(0, num_points):
        x = mean_lh[k];
        y = mean_coh[k];
        w = 2*std_lh[k];
        h = 2*std_coh[k];
        e = mpatches.Ellipse((x,y), w, h)
        e.set_alpha(0.25)
        e.set_facecolor(color)
        ax.add_patch(e);
    return ax; 


def plot_coherence_lhood(fn, means_coh, means_lh, std_coh, std_lh):
    fig, ax = plt.subplots()
    #ax = create_ellipse_mean_std(ax, , ,
    #                             , , COLORS[0]);
    plt.errorbar(means_lh['MVAE'], means_coh['MVAE'],
                 xerr=std_lh['MVAE'], yerr=std_coh['MVAE'],
                 fmt='none', mew=1.0, capsize=3.0, elinewidth=1.0)
    ax.scatter(means_lh['MVAE'], means_coh['MVAE'],
               color=COLORS[0], label='MVAE')
    #ax = create_ellipse_mean_std(ax, means_coh['MMVAE'], means_lh['MMVAE'],
    #                             std_coh['MMVAE'], std_lh['MMVAE'], COLORS[1]);
    plt.errorbar(means_lh['MMVAE'], means_coh['MMVAE'],
                 xerr=std_lh['MMVAE'], yerr=std_coh['MMVAE'],
                 fmt='none', mew=1.0, capsize=3.0, elinewidth=1.0)
    ax.scatter(means_lh['MMVAE'], means_coh['MMVAE'],
               color=COLORS[1], label='MMVAE')
    #ax = create_ellipse_mean_std(ax, means_coh['MoPoE'], means_lh['MoPoE'],
    #                             std_coh['MoPoE'], std_lh['MoPoE'], COLORS[2]);
    plt.errorbar(means_lh['MoPoE'], means_coh['MoPoE'],
                 xerr=std_lh['MoPoE'], yerr=std_coh['MoPoE'],
                 fmt='none', mew=1.0, capsize=3.0, elinewidth=1.0)
    ax.scatter(means_lh['MoPoE'], means_coh['MoPoE'],
               color=COLORS[2], label='MoPoE')
    ax.set_xlabel('Log-Likelihood', fontsize=14)
    ax.set_ylabel('Coherence Accuracy', fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.draw()
    plt.savefig(fn, format='png');



random_coh_mean_poe = [0.0408362, 0.0400328, 0.0711886, 0.12082067, 0.1819505, 0.317686]
random_coh_std_poe = [0.00281633, 0.00992883, 0.00430503, 0.0241249, 0.04952364, 0.04952364]
random_coh_mean_moe = [0.1632214, 0.1727182, 0.2530972, 0.2819524, 0.3195512, 0.3327186]
random_coh_std_moe = [0.01263696, 0.03325368, 0.01588757, 0.01026051, 0.0126645,  0.03577559]
random_coh_mean_joint = [0.112593, 0.1695422, 0.2258946, 0.3129112, 0.3892818, 0.4389736]
random_coh_std_joint = [0.01116921,  0.01459756, 0.01192365, 0.02648482, 0.0340678, 0.03781962]


lh_all_mean_poe = [-1780.5653, -1793.4125, -1778.854, -1790.0195, -1822.572, -1856.475952]
lh_all_std_poe = [3.72963061, 28.7581439, 6.96732453, 3.25776955, 1.49802764, 1.49802764]
lh_all_mean_moe = [-1951.3635, -1951.7807, -1933.9167, -1941.9084, -1953.305, -1970.2033]
lh_all_std_moe = [14.2488835, 16.3695461, 9.57064349, 5.6900855, 5.49764522, 3.27947899]
lh_all_mean_joint = [-1820.9903, -1814.1433, -1815.4845, -1819.4505, -1835.4988, -1877.6913]
lh_all_std_joint = [23.4698342, 12.4889117, 12.4481104, 5.76272754, 6.16035237, 7.77246391]

means_coh_random = {'MVAE': random_coh_mean_poe,
                    'MMVAE': random_coh_mean_moe,
                    'MoPoE': random_coh_mean_joint}
means_lh_all = {'MVAE': lh_all_mean_poe,
                'MMVAE': lh_all_mean_moe,
                'MoPoE': lh_all_mean_joint}
std_coh_random = {'MVAE': random_coh_std_poe,
                  'MMVAE': random_coh_std_moe,
                  'MoPoE': random_coh_std_joint}
std_lh_all = {'MVAE': lh_all_std_poe,
              'MMVAE': lh_all_std_moe,
              'MoPoE': lh_all_std_joint}
fn_random = os.path.join(DIR_BASE, 'figures/coherence_lhood_random.png');
plot_coherence_lhood(fn_random,
                     means_coh_random, means_lh_all,
                     std_coh_random, std_lh_all)

coh_cond_ms_t_mean_poe = [0.2392056, 0.1918114, 0.2177726, 0.29291033, 0.323649, 0.300896]
coh_cond_ms_t_std_poe = [0.08287216, 0.02662373, 0.0144425, 0.05723796, 0.09297181, 0.09297181]
coh_cond_ms_t_mean_moe = [0.8295822,  0.806697, 0.844584, 0.8367126, 0.8428678,  0.8344298]
coh_cond_ms_t_std_moe = [0.0242599, 0.09543867, 0.01957313, 0.02217157, 0.00997409, 0.00567842]
coh_cond_ms_t_mean_joint = [0.9392068, 0.9411412, 0.9338918, 0.9298716, 0.9248398, 0.9145706]
coh_cond_ms_t_std_joint = [0.00784679, 0.01022007, 0.0053768, 0.00813817, 0.0048087, 0.00156074]

lhood_cond_ms_mean_poe = [-1826.9512, -1846.0083, -1815.771, -1825.5845, -1866.3514, -1910.334717]
lhood_cond_ms_std_poe = [12.4624021, 20.0881573, 6.18014678, 2.61890866, 0.09494818, 0.09494818]
lhood_cond_ms_mean_moe = [-1923.0854, -1924.213, -1902.908, -1912.8131, -1928.575, -1951.5368]
lhood_cond_ms_std_moe = [17.8552905, 20.9441052, 12.0333773, 7.30012268, 7.25146971, 4.04700492]
lhood_cond_ms_mean_joint = [-1824.6215, -1818.4292, -1819.859, -1822.5722, -1838.0461, -1877.8747]
lhood_cond_ms_std_joint = [23.1303178, 12.0399623, 13.4291929, 4.97411109, 6.48352007, 7.12276701]

means_coh_cond_ms = {'MVAE': coh_cond_ms_t_mean_poe,
                     'MMVAE': coh_cond_ms_t_mean_moe,
                     'MoPoE': coh_cond_ms_t_mean_joint}
means_lh_cond_ms = {'MVAE': lhood_cond_ms_mean_poe,
                    'MMVAE': lhood_cond_ms_mean_moe,
                    'MoPoE': lhood_cond_ms_mean_joint}
std_coh_cond_ms = {'MVAE': coh_cond_ms_t_std_poe,
                   'MMVAE': coh_cond_ms_t_std_moe,
                   'MoPoE': coh_cond_ms_t_std_joint}
std_lh_cond_ms = {'MVAE': lhood_cond_ms_std_poe,
                  'MMVAE': lhood_cond_ms_std_moe,
                  'MoPoE': lhood_cond_ms_std_joint}
fn_cond_ms = os.path.join(DIR_BASE, 'figures/coherence_lhood_ms_t.png');
plot_coherence_lhood(fn_cond_ms,
                     means_coh_cond_ms, means_lh_cond_ms,
                     std_coh_cond_ms, std_lh_cond_ms)

lhood_cond_mt_mean_poe = [-2079.6371, -2052.9737, -2055.7776, -2050.3156, -2042.3918, -2038.439941]
lhood_cond_mt_std_poe = [7.49229389, 35.796109, 5.16513655, 3.72185102, 0.63986588, 0.63986588]
lhood_cond_mt_mean_moe = [-2014.6277, -2009.5432, -2003.0596, -2002.4356, -2002.979, -2009.3535]
lhood_cond_mt_std_moe = [3.79184026, 2.06106217, 0.42810588, 1.16638784, 0.90121596, 0.89580292]
lhood_cond_mt_mean_joint = [-2008.7739, -1994.2135, -1986.0623, -1987.6719, -1990.8458, -1998.5002]
lhood_cond_mt_std_joint = [7.1628989, 4.22490031, 2.52879341, 3.06277149, 1.30928024, 2.5746049]

coh_cond_mt_s_mean_poe = [0.8116968, 0.824856, 0.7852692, 0.745015, 0.8567265, 0.842825]
coh_cond_mt_s_std_poe = [0.01887247, 0.03920606, 0.01832331, 0.04127158, 0.01199324, 0.01199324]
coh_cond_mt_s_mean_moe = [0.3669402, 0.2762194, 0.3453684, 0.302941, 0.2916074, 0.294028]
coh_cond_mt_s_mean_moe = [0.02055449, 0.12156079, 0.06299504, 0.03364374, 0.01952061, 0.04845713]
coh_cond_mt_s_std_moe = [0.02055449, 0.12156079, 0.06299504, 0.03364374, 0.01952061, 0.04845713]
coh_cond_mt_s_mean_joint = [0.3087238, 0.3097082, 0.2913942, 0.3716656, 0.5604074, 0.6159962]
coh_cond_mt_s_std_joint = [0.07107001, 0.03452847, 0.03366899, 0.06719986, 0.04050825, 0.09025252]

means_coh_cond_mt = {'MVAE': coh_cond_mt_s_mean_poe,
                     'MMVAE': coh_cond_mt_s_mean_moe,
                     'MoPoE': coh_cond_mt_s_mean_joint}
means_lh_cond_mt = {'MVAE': lhood_cond_mt_mean_poe,
                    'MMVAE': lhood_cond_mt_mean_moe,
                    'MoPoE': lhood_cond_mt_mean_joint}
std_coh_cond_mt = {'MVAE': coh_cond_mt_s_std_poe,
                   'MMVAE': coh_cond_mt_s_std_moe,
                   'MoPoE': coh_cond_mt_s_std_joint}
std_lh_cond_mt = {'MVAE': lhood_cond_mt_std_poe,
                  'MMVAE': lhood_cond_mt_std_moe,
                  'MoPoE': lhood_cond_mt_std_joint}
fn_cond_mt = os.path.join(DIR_BASE, 'figures/coherence_lhood_mt_s.png');
plot_coherence_lhood(fn_cond_mt,
                     means_coh_cond_mt, means_lh_cond_mt,
                     std_coh_cond_mt, std_lh_cond_mt)

lhood_cond_st_mean_poe = [-1839.7227, -1869.6724, -1842.5361, -1855.5498, -1872.7409, -1892.700928]
lhood_cond_st_std_poe = [23.8533865, 25.8916347, 8.90811534, 0.25404237, 5.0057673, 5.0057673]
lhood_cond_st_mean_moe = [-1933.3305, -1936.641, -1915.3532, -1925.2973, -1939.3296, -1956.598]
lhood_cond_st_std_moe = [17.5565234, 20.6546658, 10.4451566, 7.66553982, 6.95292358, 4.06396573]
lhood_cond_st_mean_joint = [-1852.3204, -1845.1026, -1848.6349, -1850.6511, -1862.1164, -1892.4623]
lhood_cond_st_std_joint = [23.3181933, 11.472898, 11.5258971, 5.81934681, 7.51044594, 7.46309189]

coh_cond_st_m_mean_poe = [0.100708, 0.1213598, 0.2209196, 0.324622, 0.4676745, 0.510298]
coh_cond_st_m_std_poe = [0.00678435, 0.0153438, 0.03057503, 0.02701291, 0.21405324, 0.21405324]
coh_cond_st_m_mean_moe = [0.8590598, 0.8322594, 0.8827574, 0.8706398, 0.8772176, 0.8685398]
coh_cond_st_m_std_moe = [0.02794791, 0.11049504, 0.02298551, 0.02903281, 0.00992234, 0.00720535]
coh_cond_st_m_mean_joint = [0.9520874, 0.9490568, 0.9426368, 0.937577, 0.928248, 0.9077454]
coh_cond_st_m_std_joint = [0.00906728, 0.01587776, 0.01107054, 0.00669708, 0.01094543, 0.0377521]

means_coh_cond_st = {'MVAE': coh_cond_st_m_mean_poe,
                     'MMVAE': coh_cond_st_m_mean_moe,
                     'MoPoE': coh_cond_st_m_mean_joint}
means_lh_cond_st = {'MVAE': lhood_cond_st_mean_poe,
                    'MMVAE': lhood_cond_st_mean_moe,
                    'MoPoE': lhood_cond_st_mean_joint}
std_coh_cond_st = {'MVAE': coh_cond_st_m_std_poe,
                   'MMVAE': coh_cond_st_m_std_moe,
                   'MoPoE': coh_cond_st_m_std_joint}
std_lh_cond_st = {'MVAE': lhood_cond_st_std_poe,
                  'MMVAE': lhood_cond_st_std_moe,
                  'MoPoE': lhood_cond_st_std_joint}
fn_cond_st = os.path.join(DIR_BASE, 'figures/coherence_lhood_st_m.png');
plot_coherence_lhood(fn_cond_st,
                     means_coh_cond_st, means_lh_cond_st,
                     std_coh_cond_st, std_lh_cond_st)

lhood_cond_m_mean_poe = [-2119.7954, -2092.7867, -2096.2627, -2090.1691, -2086.7239, -2094.216553]
lhood_cond_m_std_poe = [10.1194854, 37.9834473, 7.30625258, 3.75560172, 0.09702, 0.09702]
lhood_cond_m_mean_moe = [-2002.0694, -1993.9265, -1987.1472, -1987.0894, -1990.3073, -2003.0612]
lhood_cond_m_std_moe = [3.76059021, 1.76716696, 1.47723691, 1.45210978, 1.66639112, 1.42097466]
lhood_cond_m_mean_joint = [-2017.3432, -2002.1033, -1990.7746, -1991.4939, -1996.0791, -2001.0868]
lhood_cond_m_std_joint = [13.9231091, 4.82600781, 4.44717176, 2.89131292, 1.30358744, 2.40355582]

coh_cond_m_s_mean_poe = [0.4512754, 0.2375648, 0.3376672, 0.430016, 0.2714365, 0.23504]
coh_cond_m_s_std_poe = [0.0892608, 0.03584278, 0.04546458, 0.02123037, 0.03736847, 0.03736847]
coh_cond_m_s_mean_moe = [0.368722, 0.2769614, 0.3338578, 0.3052746, 0.3052318, 0.3383434]
coh_cond_m_s_std_moe = [0.02138366, 0.12209307, 0.07505911, 0.02802714, 0.02205423, 0.06758739]
coh_cond_m_s_mean_joint = [0.2878832, 0.2775558, 0.2751922, 0.3604014, 0.4708246, 0.5496388]
coh_cond_m_s_std_joint = [0.07187711, 0.04106908, 0.03060899, 0.071012, 0.04502212, 0.05953834]

coh_cond_m_t_mean_poe = [0.2906062, 0.1949576, 0.1830516, 0.280493, 0.3361675, 0.301802]
coh_cond_m_t_std_poe = [0.10338974, 0.0236923, 0.00537939, 0.05640505, 0.10319304, 0.10319304]
coh_cond_m_t_mean_moe = [0.965973, 0.9661722, 0.95593, 0.956421, 0.9534692, 0.94618]
coh_cond_m_t_std_moe = [0.00135708, 0.00102696, 0.0118912, 0.0015934, 0.00172734, 0.00218508]
coh_cond_m_t_mean_joint = [0.9649988, 0.9625522, 0.9595182, 0.9542124, 0.950361, 0.93783]
coh_cond_m_t_std_joint = [0.00338937, 0.00260731, 0.0029692, 0.00273139, 0.00346956, 0.00544863]

means_lh_cond_m = {'MVAE': lhood_cond_m_mean_poe,
                   'MMVAE': lhood_cond_m_mean_moe,
                   'MoPoE': lhood_cond_m_mean_joint}
std_lh_cond_m = {'MVAE': lhood_cond_m_std_poe,
                   'MMVAE': lhood_cond_m_std_moe,
                   'MoPoE': lhood_cond_m_std_joint}
means_coh_cond_m_s = {'MVAE': coh_cond_m_s_mean_poe,
                     'MMVAE': coh_cond_m_s_mean_moe,
                     'MoPoE': coh_cond_m_s_mean_joint}
std_coh_cond_m_s = {'MVAE': coh_cond_m_s_std_poe,
                    'MMVAE': coh_cond_m_s_std_moe,
                    'MoPoE': coh_cond_m_s_std_joint}
means_coh_cond_m_t = {'MVAE': coh_cond_m_t_mean_poe,
                     'MMVAE': coh_cond_m_t_mean_moe,
                     'MoPoE': coh_cond_m_t_mean_joint}
std_coh_cond_m_t = {'MVAE': coh_cond_m_s_std_poe,
                    'MMVAE': coh_cond_m_s_std_moe,
                    'MoPoE': coh_cond_m_s_std_joint}

fn_cond_m_s = os.path.join(DIR_BASE, 'figures/coherence_lhood_m_s.png');
plot_coherence_lhood(fn_cond_m_s,
                     means_coh_cond_m_s, means_lh_cond_m,
                     std_coh_cond_m_s, std_lh_cond_m)
fn_cond_m_t = os.path.join(DIR_BASE, 'figures/coherence_lhood_m_t.png');
plot_coherence_lhood(fn_cond_m_t,
                     means_coh_cond_m_t, means_lh_cond_m,
                     std_coh_cond_m_t, std_lh_cond_m)


lhood_cond_s_mean_poe = [-1865.7548, -1894.2813, -1875.4211, -1895.4515, -1925.4725, -1951.84668]
lhood_cond_s_std_poe = [16.5703495, 28.4855985, 5.83516166, 0.18746229, 0.28579488, 0.28579488]
lhood_cond_s_mean_moe = [-1864.2539, -1871.4355, -1840.7209, -1856.9876, -1880.5691, -1909.5842]
lhood_cond_s_std_moe = [27.3250905, 33.6099765, 16.2563859, 12.4184697, 11.1584166, 6.52641256]
lhood_cond_s_mean_joint = [-1865.3464, -1854.0601, -1858.7157, -1858.1226, -1872.3079, -1902.1936]
lhood_cond_s_std_joint = [25.7032091,17.2243829, 13.7119402, 6.16848551, 10.3958703,7.73706866]

coh_cond_s_m_mean_poe = [0.099518, 0.1109354, 0.2063472, 0.238663, 0.118223, 0.099942]
coh_cond_s_m_std_poe = [0.00743685, 0.01060265, 0.03105124, 0.01269422, 9.19E-05, 9.19E-05]
coh_cond_s_m_mean_moe = [0.719053, 0.6650506, 0.7662072, 0.7464772, 0.7631344, 0.7529678]
coh_cond_s_m_std_moe = [0.05513254, 0.22212903, 0.04599726, 0.057367, 0.01976647, 0.00780424]
coh_cond_s_m_mean_joint = [0.7357804, 0.762566, 0.7297346, 0.7406838, 0.6952784, 0.727853]
coh_cond_s_m_std_joint = [0.07974815, 0.04276403, 0.01990782, 0.03512292, 0.07385106, 0.01608564]

coh_cond_s_t_mean_poe = [0.1095712, 0.1050406, 0.1632954, 0.16679667, 0.114897, 0.099187]
coh_cond_s_t_std_poe = [0.00631291, 0.00285995, 0.01060668, 0.01807168, 0.0051138, 0.0051138]
coh_cond_s_t_mean_moe = [0.7380504, 0.6942902, 0.7772278, 0.7616068, 0.7754652, 0.7662574]
coh_cond_s_t_std_moe = [0.04725758, 0.18511886, 0.04299407, 0.04344023, 0.01998618, 0.00897755]
coh_cond_s_t_mean_joint = [0.7631992, 0.7836368, 0.7539762, 0.7625362, 0.7418426, 0.7607864]
coh_cond_s_t_std_joint = [0.06348377, 0.03323282, 0.01679986, 0.0316808, 0.04735723, 0.01268]

means_lh_cond_s = {'MVAE': lhood_cond_s_mean_poe,
                   'MMVAE': lhood_cond_s_mean_moe,
                   'MoPoE': lhood_cond_s_mean_joint}
std_lh_cond_s = {'MVAE': lhood_cond_s_std_poe,
                   'MMVAE': lhood_cond_s_std_moe,
                   'MoPoE': lhood_cond_s_std_joint}
means_coh_cond_s_m = {'MVAE': coh_cond_s_m_mean_poe,
                     'MMVAE': coh_cond_s_m_mean_moe,
                     'MoPoE': coh_cond_s_m_mean_joint}
std_coh_cond_s_m = {'MVAE': coh_cond_s_m_std_poe,
                    'MMVAE': coh_cond_s_m_std_moe,
                    'MoPoE': coh_cond_s_m_std_joint}
means_coh_cond_s_t = {'MVAE': coh_cond_s_t_mean_poe,
                     'MMVAE': coh_cond_s_t_mean_moe,
                     'MoPoE': coh_cond_s_t_mean_joint}
std_coh_cond_s_t = {'MVAE': coh_cond_s_t_std_poe,
                    'MMVAE': coh_cond_s_t_std_moe,
                    'MoPoE': coh_cond_s_t_std_joint}
fn_cond_s_m = os.path.join(DIR_BASE, 'figures/coherence_lhood_s_m.png');
plot_coherence_lhood(fn_cond_s_m,
                     means_coh_cond_s_m, means_lh_cond_s,
                     std_coh_cond_s_m, std_lh_cond_s)

fn_cond_s_t = os.path.join(DIR_BASE, 'figures/coherence_lhood_s_t.png');
plot_coherence_lhood(fn_cond_s_t,
                     means_coh_cond_s_t, means_lh_cond_s,
                     std_coh_cond_s_t, std_lh_cond_s)


lhood_cond_t_mean_poe = [-2161.4194, -2143.8618, -2149.0738, -2133.6252, -2095.2188, -2074.92041]
lhood_cond_t_std_poe = [6.45068015, 35.510948, 6.01780002, 6.87305841, 6.91742341, 6.91742341]
lhood_cond_t_mean_moe = [-2027.6392, -2025.76, -2019.8753, -2018.2806, -2016.0459, -2015.7227]
lhood_cond_t_std_moe = [4.86854325, 3.98531416, 1.43478944, 1.61330261, 0.30205471, 0.59518195]
lhood_cond_t_mean_joint = [-2040.8881, -2029.3015, -2024.3432, -2024.4229, -2023.3356, -2019.9527]
lhood_cond_t_std_joint = [6.7548794, 2.17155239, 1.62353098, 2.55812245, 1.07178124, 1.22135568]

coh_cond_t_m_mean_poe = [0.128238, 0.1179438, 0.1216288, 0.19615267, 0.472091, 0.508588]
coh_cond_t_m_std_poe = [0.0006, 0.00961662, 0.00426535, 0.04956035, 0.2402664, 0.2402664]
coh_cond_t_m_mean_moe = [0.999691, 0.999086, 0.9988496, 0.9942552, 0.9921166, 0.984417]
coh_cond_t_m_std_moe = [4.29E-05, 0.0004521, 0.00057642, 0.00346476, 0.00249825, 0.01665809]
coh_cond_t_m_mean_joint = [0.9940602, 0.9984396, 0.978545, 0.9904268, 0.9812246, 0.9615826]
coh_cond_t_m_std_joint = [0.01036713, 0.00199201, 0.03988991, 0.01052866, 0.00643075, 0.04252222]

coh_cond_t_s_mean_poe = [0.2989436, 0.4199324, 0.271719, 0.301303, 0.5977835, 0.5312]
coh_cond_t_s_std_poe = [0.08009603, 0.12298843, 0.03079166, 0.08192542, 0.17275538, 0.17275538]
coh_cond_t_s_mean_moe = [0.36565, 0.2762434, 0.3601492, 0.2973744, 0.2767458, 0.2483556]
coh_cond_t_s_std_moe = [0.01972038, 0.12240518, 0.04938006, 0.04095373, 0.02488131, 0.04169571]
coh_cond_t_s_mean_joint = [0.3123048, 0.2992346, 0.2771216, 0.3392366, 0.405922, 0.3911804]
coh_cond_t_s_std_joint = [0.07243515, 0.0216446, 0.03316877, 0.06566128, 0.08153717, 0.05253327]

means_lh_cond_t = {'MVAE': lhood_cond_t_mean_poe,
                   'MMVAE': lhood_cond_t_mean_moe,
                   'MoPoE': lhood_cond_t_mean_joint}
std_lh_cond_t = {'MVAE': lhood_cond_t_std_poe,
                   'MMVAE': lhood_cond_t_std_moe,
                   'MoPoE': lhood_cond_t_std_joint}
means_coh_cond_t_m = {'MVAE': coh_cond_t_m_mean_poe,
                     'MMVAE': coh_cond_t_m_mean_moe,
                     'MoPoE': coh_cond_t_m_mean_joint}
std_coh_cond_t_m = {'MVAE': coh_cond_t_m_std_poe,
                    'MMVAE': coh_cond_t_m_std_moe,
                    'MoPoE': coh_cond_t_m_std_joint}
means_coh_cond_t_s = {'MVAE': coh_cond_t_s_mean_poe,
                     'MMVAE': coh_cond_t_s_mean_moe,
                     'MoPoE': coh_cond_t_s_mean_joint}
std_coh_cond_t_s = {'MVAE': coh_cond_t_s_std_poe,
                    'MMVAE': coh_cond_t_s_std_moe,
                    'MoPoE': coh_cond_t_s_std_joint}
fn_cond_t_m = os.path.join(DIR_BASE, 'figures/coherence_lhood_t_m.png');
plot_coherence_lhood(fn_cond_t_m,
                     means_coh_cond_t_m, means_lh_cond_t,
                     std_coh_cond_t_m, std_lh_cond_t)
fn_cond_t_s = os.path.join(DIR_BASE, 'figures/coherence_lhood_t_s.png');
plot_coherence_lhood(fn_cond_t_s,
                     means_coh_cond_t_s, means_lh_cond_t,
                     std_coh_cond_t_s, std_lh_cond_t)
