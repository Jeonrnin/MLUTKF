import os
import pickle

def set_plot_font(plt,i):
    """
    i=0: Arial
    """
    if i == 0:                                          # Arial
        plt.rc('font', family='Arial')                  
        #plt.rc('mathtext', fontset="cm")               # math text: Times New Roman 
        plt.rc('mathtext', fontset="custom")
        plt.rc('mathtext', rm="Arial")                  # math text: Arial 

def save_plot(plt,fdir,ext_list):
    f = os.path.splitext(fdir)[0]

    for ext in ext_list:        
        plt.savefig(os.path.join('{}.{}'.format(f,ext)), bbox_inches='tight', format=ext)

def save_plot_data_by_pickle(fdir,data):
    f = open(fdir, 'wb')
    pickle.dump(data, f)
    f.close()

def load_plot_data_by_pickle(fdir):
    f = open(fdir, 'rb')
    data = pkl_file(f)
    f.close()

    return data
    
def h_mathexp_str(i):
    """
    i=0: h(x)=x
    i=1: h(x)=|x|
    i=2: h(x)=ln(|x|)
    """
    h_types = ['$h\mathrm{(x)=x}$', '$h\mathrm{(x)=|x|}$', '$h\mathrm{(x)=ln(|x|)}$']
    
    return h_types[i]         
