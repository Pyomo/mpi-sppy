import os
import json
import mpisppy.extensions.extension

class DatahashCollector(mpisppy.extensions.extension.Extension):
    '''
    This class is implemented as an extension to be used in mpi-sppy to collect the data hash
    of all scenarios and write it to a json file.

    This is useful when running multiple tests on variations of the same model to make sure that the
    analysis is performed on the same test case that was used to obtain the results.

    A file called data_hash.json is created with a dict with the following format
    {cep_data_hash: cep_data.data_hash(),
     cep_timevar_data_hash: {scen_name: cep_timevar_data.data_hash() for each scen_name}
    }

    Attributes
    ----------
    output_dir: directory where the output file is saved
    '''

    def __init__(self, ph):

        self.ph = ph

        try:
            self.output_dir = ph.options['datahash_collector']['output_dir']
        except KeyError:
            print('Must specify an output directory to use the Datahash Collector extension')
            raise
            

    def post_everything(self):
        ph = self.ph
        cep_data_flag = True

        comm = ph.mpicomm
        
        local_data_hashes = dict()
        for sname, s in ph.local_scenarios.items():
            # Get cep_data hash (only once)
            if ph.cylinder_rank==0 and cep_data_flag:
                cep_data_hash = s.data.data_hash()
                cep_data_flag = False

            # Get cep_timevar hash
            local_data_hashes[sname]=s.timevar_data.data_hash()

        all_data_hashes = comm.gather(
            local_data_hashes, root=0
        )

        datahash_dict = dict()

        if ph.cylinder_rank != 0:
            return
        datahash_dict['cep_data_hash']=cep_data_hash
        datahash_dict['cep_timevar_data_hash'] = dict()
        for dh_d in all_data_hashes:
            datahash_dict['cep_timevar_data_hash'].update(dh_d)

        with open(os.path.join(self.output_dir,'data_hash.json'),'w') as f:
            json.dump(datahash_dict,f)
            
        



