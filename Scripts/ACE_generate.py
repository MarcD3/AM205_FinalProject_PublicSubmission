import numpy as np
from  ase.io import read, write
from flare.bffs.sgp._C_flare import Structure, B2
import os

class Generate_ACE():
    
    def __init__(self, species_map, label, store_folder):
        self.species_map = species_map
        self.label = label
        self.store_folder = store_folder

    def ACE_descriptor(self, frame, cutoff, N_max, L_max, radial_basis, radial_hyps):
        n_species = len(self.species_map)
        cutoff_name = "quadratic"
        cutoff_hyps = []
        settings = [n_species, N_max, L_max]
        calc = B2(radial_basis, cutoff_name, radial_hyps, cutoff_hyps, settings)
        calc_list = [calc] # multi descriptors here

        species = [self.species_map[spec] for spec in frame.numbers]
        
        # compute my descriptors (here use B2 as an example)
        struc = Structure(
            frame.cell,  # atoms.cell,
            species,
            frame.positions, # atoms.positions
            cutoff,
            calc_list,
        )
        mydesc = struc.descriptors[0].descriptors

        desc = mydesc[0]
        for i in np.arange(1, n_species):
            desc = np.append(desc, mydesc[i] ,axis=0)

        norm = np.linalg.norm(desc, axis=1, keepdims=True)
        mydesc_normed = desc/norm # (N x d)

        self.desc = desc
        self.mydesc_normed = mydesc_normed
        return desc, mydesc_normed

    
    def store(self, name, N_max, L_max, cutoff):

        label_arr = np.ones((self.desc.shape))*self.label
        
        name1 = '%s/%s_Nmax_%s_Lmax_%s_cutoff_%s.npz'%(self.store_folder, name,  N_max, L_max, cutoff)
        name2 =  '%s/norm_%s_Nmax_%s_Lmax_%s_cutoff_%s.npz'%(self.store_folder, name, N_max, L_max, cutoff)
        np.savez_compressed(name1,ACE=self.desc,label=label_arr)
        np.savez_compressed(name2,ACE=self.mydesc_normed,label=label_arr)

    def get_multi_frame(self, data, cutoff, N_max, L_max, radial_basis, radial_hyps, row_range = slice(None)):
        descs = []
        desc_norms = []
        for i in range(0,len(data),1):
            desc, desc_norm = self.ACE_descriptor(data[i], cutoff, N_max, L_max, radial_basis, radial_hyps)
            descs.append(desc[row_range,:])
            desc_norms.append(desc_norm[row_range,:])
        
        descs = np.vstack(np.array(descs))
        desc_norms = np.vstack(np.array(desc_norms))

        return descs, desc_norms
    
    def get_interface(self, folder, store_folder, cutoff, N_max, L_max, radial_basis, radial_hyps):
        desc_all = []
        labels_all = []
        desc_norm_all = []

        row_range = {'Li3P': slice(0, 6), 'LiCl': slice(0, 4), 'bulkLi': slice(None), 'LPSCl': slice(0, 24), 'Li2S': slice(0,8)}
        label_dict = {'bulkLi': 0, 'LPSCl': 1, 'LiCl': 2, 'Li2S': 3, 'Li3P': 4}

        files = [file for file in os.listdir(folder) if (file.endswith('.xyz'))]

        for file in files:
            file_name = os.path.join(folder, file)

            # get the type
            type = file.split('.')[0].split('_')[0]

            data = read(file_name, index=':')
            desc, desc_norm = self.get_multi_frame(data, cutoff, N_max, L_max, radial_basis, radial_hyps, row_range[type])

            desc_all.append(list(desc))
            desc_norm_all.append(list(desc_norm))

            # create labels
            label_array = np.ones(desc.shape[0])*label_dict[type]
            labels_all.append(label_array)
        
        desc_all = np.vstack((desc_all[0], desc_all[1], desc_all[2], desc_all[3], desc_all[4]))
        desc_norm_all = np.vstack((desc_norm_all[0], desc_norm_all[1], desc_norm_all[2], desc_norm_all[3], desc_norm_all[4]))
        labels_all = np.concatenate((labels_all[0], labels_all[1], labels_all[2], labels_all[3], labels_all[4]))

        #name1 = '%s/n_max_%s_l_max_%s_cutoff_%s.npz'%(store_folder, N_max, L_max, cutoff)
        name2 =  '%s/norm_n_max_%s_l_max_%s_cutoff_%s_basis_%s.npz'%(store_folder, N_max, L_max, cutoff, radial_basis)
        #np.savez_compressed(name1, ACE=desc_all, label=labels_all)
        np.savez_compressed(name2, ACE=desc_norm_all, label=labels_all)

# folder_data = '/n/holystore01/LABS/kozinsky_lab/Lab/User/mdescoteaux/Projects/23_11_17_AM205/Structures/'

folder_data = '/n/holystore01/LABS/kozinsky_lab/Lab/User/classifier_data/structures/'

# for f in [file for file in os.listdir(folder_data) if (file.endswith('.xyz'))]:
for i in [1]:
    # filename = os.path.join(folder_data, f)
    # mol = f.split('.')[0]

    # data = read(filename,index=':')
    # type = f.split('.')[0].split('_')[-1]

    # species inside file and their labels
    # species_map = {}
    # if(type == 'H2O'):
    #     species_map = {8: 0, 1:1}
    # else:
    #     species_map = {8:0, 6:1}

    # label = 0
    species_map = {3: 0, 16: 1, 15: 2, 17: 3}

    # where the ACE should be stored
    store_folder = 'ACE/interface_new'

    ace = Generate_ACE(species_map, None, store_folder)

    # generate the ACE descriptor
   # for radial_basis in ['weighted_chebyshev','positive_chebyshev','weighted_positive_chebyshev', 'poly', 'fourier_quarter', 'fourier_half', 'fourier', 'equispaced_gaussians']:
    for radial_basis in ['chebyshev']:
        for cutoff in np.arange(4, 12, 2):
            for N_max in np.arange(2, 12, 2):
                for L_max in np.arange(2, 12, 2):

                    radial_hyps = [0, cutoff]

                    if(radial_basis == 'equispaced_gaussians'):
                        radial_hyps = [0, cutoff, 1]
                    elif(radial_basis == 'fourier'):
                        radial_hyps = [0,cutoff, 2.0]
                    elif(radial_basis == "weighted_chebyshev"):
                        radial_hyps = [0,cutoff, 5.0]
                    elif(radial_basis == "weighted_positive_chebyshev"):
                        radial_hyps = [0,cutoff, 5.0]


                    # ace.ACE_descriptor(data[0], cutoff, N_max, L_max, radial_basis, radial_hyps)

                    # # save the ACE descriptor
                    # radial_hyps_str = ''
                    # for param in radial_hyps:
                    #     radial_hyps_str = radial_hyps_str + '_' + str(param)
                    # name = '%s_%s%s'%(mol, radial_basis, radial_hyps_str)
                    # ace.store(name, N_max, L_max, cutoff)

                    ## interface
                    ace.get_interface(folder_data, store_folder, cutoff, N_max, L_max, radial_basis, radial_hyps)


        
