import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors
import mplhep as hep
import matplotlib
plt.style.use(hep.style.ROOT)
matplotlib.rcParams.update({'font.size': 20})

class PCA_plot:

    def __init__(self, ace, label, title, label_dict):

        # PCA
        self.pca = PCA(n_components=2)
        
        self.label_dict = label_dict

        # fit on training data
        self.ACE_pca = self.pca.fit_transform(ace)

        # plot on training
        self.pca_plot(self.ACE_pca, label, title)


    def pca_plot(self, pca_in, label, title):

        colors = ['#762a83', '#af8dc3', '#e7d4e8', '#7fbf7b', '#1b7837' ]
        plt.figure(figsize=(8, 6))
       
        if(np.any(label == None)):
            plt.scatter(pca_in[:,0], pca_in[:, 1])
        else:
            for l_str, c in zip (self.label_dict.keys(), colors):
                mask = label == self.label_dict[l_str]
                plt.scatter(pca_in[mask, 0], pca_in[mask, 1], label=l_str, color=c)
        if(np.all(label != None)):
            plt.legend()
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig('PCA/interface/%s.png'%(title))
        plt.close()


def run_pca_interface():
    folder_data = 'ACE/interface/'
    label_dict = {'bulkLi': 0, 'LPSCl': 1, 'LiCl': 2, 'Li2S': 3, 'Li3P': 4}
    store_folder = 'ACE/interface'

    for radial_basis in ['chebyshev']:
        for cutoff in np.arange(4, 12, 2):
            for N_max in np.arange(2, 12, 2):
                for L_max in np.arange(2, 12, 2):

                    filename = '%s/norm_n_max_%s_l_max_%s_cutoff_%s_basis_%s.npz'%(store_folder, N_max, L_max, cutoff, radial_basis)
                    print(filename)
                    data = np.load(filename)
                    if(data['ACE'][0][0] == data['ACE'][0][0]):
                        pca = PCA_plot(data['ACE'], data['label'], filename.split('.')[0].split('interface/')[-1], label_dict)

def pca_variance_heatmap(cutoff, radial_basis):

    Nmax = np.arange(2, 12, 2)
    Lmax = np.arange(2, 12, 2)
    data_folder = 'ACE/interface_new'
    store_folder = 'PCA/heatmap'


    var_mat = np.zeros((Nmax.size, Lmax.size))

    for row_index, n in enumerate(Nmax):
        for col_index, l in enumerate(Lmax):
        
            filename = '%s/norm_n_max_%s_l_max_%s_cutoff_%s_basis_%s.npz'%(data_folder, n, l, cutoff, radial_basis)
            data =  np.load(filename)
            pca = PCA(n_components=2)
            model = pca.fit_transform(data['ACE'])
            var_mat[row_index][col_index] = pca.explained_variance_ratio_.cumsum()[-1]
    
    print(var_mat[:10])
    # save the array for the heatmap
    name =  '%s/norm_cutoff_%s_basis_%s.npy'%(store_folder, cutoff, radial_basis)
        #np.savez_compressed(name1, ACE=desc_all, label=labels_all)
    np.save(name, var_mat)

def combine_files(folder, type_atom, basis, type_change, N_max = 10, L_max = 3):

    files_1 = [file for file in os.listdir('%s/%s'%(folder, 'ACE')) if (('%s_%s_0'%(type_atom, basis) in file) and ('Nmax_%s'%(N_max) in file) and ('Lmax_%s'%(N_max) in file) and ('norm' in file) and (type_change in file) and not('default' in file))]
    # files_2 = [file for file in os.listdir('%s/%s'%(folder, 'ACE_poly')) if (('%s_%s'%(type_atom, basis) in file) and ('norm' in file) and (type_change in file))]
    
    changes = []
    ACE = []
    for file in files_1:
        changes.append(float(file.split('_')[3])/1000)

        # get ACE
        full_file = os.path.join('%s/%s'%(folder, 'ACE'), file)
        ACE.append(np.load(full_file)['ACE'][0])

    print("number of files: ", len(changes))
    
    return changes, np.array(ACE)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_colors(array, map):
    colors = []
    for val in array:
        colors.append(map(val))
    return colors

def create_PCA(store_folder, folder, func, type_change):

    # combine all h20 and co2
    bend_h20, ace_h20 = combine_files(folder, 'H2O', func, type_change)
    bend_co2, ace_co2 = combine_files(folder, 'CO2', func, type_change)
    
    # create new heatmaps for colors 
    cmap = plt.get_cmap('Greens')
    new_greens = truncate_colormap(cmap, 0.4, 1)

    cmap = plt.get_cmap('Purples')
    new_purples = truncate_colormap(cmap, 0.55, 1)

    # get colors
    colors_h20 = get_colors(bend_h20, new_purples)
    colors_co2 = get_colors(bend_co2, new_greens)

    # combine h20 and c02
    all = np.vstack((ace_h20, ace_co2))

    all_colors = np.vstack((colors_h20, colors_co2))

    pca = PCA(n_components=2)
    ACE_pca = pca.fit_transform(all)

    # save colors and pca
    name =  '%s/%s_%s.npz'%(store_folder, func, type_change)

    np.savez_compressed(name, ACE=all, colors=all_colors)

# folder = '/n/holystore01/LABS/kozinsky_lab/Lab/User/mdescoteaux/Projects/23_11_17_AM205/AceDescriptor_v2'
# for type_change in ['bond_delta', 'alignment_angle']:
#     store_folder = 'PCA/ACE/%s'%(type_change)
#     for basis in ['chebyshev', 'positive_chebyshev', 'fourier', 'fourier_quarter']:
#         create_PCA(store_folder, folder, basis, type_change)

run_pca_interface()
# for cutoff in [4, 6, 8]:
#    # for basis in ['weighted_chebyshev','positive_chebyshev','weighted_positive_chebyshev', 'poly', 'fourier_quarter', 'fourier_half', 'fourier', 'equispaced_gaussians']:
#     for basis in ['chebyshev']:
#         pca_variance_heatmap(cutoff, basis)
