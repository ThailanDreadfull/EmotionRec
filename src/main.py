#import numpy as np
#from hmmlearn import hmm
#np.random.seed(42)
#
#model = hmm.GaussianHMM(n_components=3, covariance_type="full")
#model.startprob_ = np.array([0.6, 0.3, 0.1])
#model.transmat_ = np.array([[0.7, 0.2, 0.1],
#                            [0.3, 0.5, 0.2],
#                            [0.3, 0.3, 0.4]])

#model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
#model.covars_ = np.tile(np.identity(2), (3, 1, 1))
#X, Z = model.sample(100)


#lr = hmm.GaussianHMM(n_components=3, covariance_type="diag", init_params="cm", params="cmt")
#lr.startprob_ = np.array([1.0, 0.0, 0.0])
#lr.transmat_ = np.array([[0.5, 0.5, 0.0],
#                          [0.0, 0.5, 0.5],
#                          [0.0, 0.0, 1.0]])
#





import subprocess
from bregman.suite import Chromagram








def extract_chromagram(audio_file_path):
    
    
    F = Chromagram(audio_file_path, nfft=16384, wfft=8192, nhop=2205)
    all_chroma_ft = F.X # all chroma features
    one_ft = F.X[:,0] # one feature
    
    print('All chrome features: '+ str(all_chroma_ft))
    print('One feature: '+ str(one_ft))





def extract_parameters():
    cmd = []
    cmd.append('yaafe')
    cmd.append('-v')
    cmd.append('-r')
    cmd.append('44100')
    cmd.append('-o')
    cmd.append('csv')
    cmd.append('-i')
    cmd.append('/root/tcc/extraction/to_extract')
    cmd.append('-b')
    cmd.append('/root/tcc/extraction/output')
    #cmd.append('-f')
    #cmd.append('Energy:Derivate')
    cmd.append('-c')
    cmd.append('/root/tcc/extraction/featurePlan')
    
    subprocess.check_output(cmd)
    
    print("Command: "+" ".join(cmd))    

    output, err  = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()

    if not err:
        print('>>>>>>>>>>>>>>>>>>>>>>>>')
        print(output)
        print('>>>>>>>>>>>>>>>>>>>>>>>>')
    else:
        print('>>>>>>>>>>> error <<<<<<<<<<<<')
        print(err)
        print('>>>>>>>>>>>>>><<<<<<<<<<<<<<<<')



if __name__ == '__main__':
    print('Starting program...')
    #extract_parameters()
    
    extract_chromagram("/root/tcc/extraction/audio_samples/teste2-2.wav")
    
    print('Ending program...')


