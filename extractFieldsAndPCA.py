import sys, os
sys.path.append('/Users/cequilod/')
import vtktools
import numpy as np
import pyvista as pv
import eofs
from eofs.standard import Eof
from variables import directory_data, field_name, start, end, nsize

class extractFieldsAndPCA():
    def __init__(self, directory_data, start, end, start_pca, end_pca, nsize):

        self.directory_data = directory_data
        self.start = start
        self.end = end
        self.nsize = nsize

        #For PCA analysis specific time-steps
        self.start_pca = start_pca
        self.end_pca = end_pca

    def extractFields(self):
        tracer_data = np.zeros((self.end-self.start, self.nsize))
        vel_data = np.zeros((self.end-self.start, self.nsize, 3))
        k = 0
        for i in np.arange(self.start, self.end):
                filename = self.directory_data + 'LSBU_' + str(i) + '.vtu'
                mesh = pv.read(filename)
                tracer_data[k, :] = np.squeeze(mesh.get_array('Tracer'))
                vel_data[k, :, :] = mesh.get_array('Velocity')
                print(k)
                k = k + 1

        np.save(self.directory_data + 'Tracer_data_' + str(self.start) + '_to_' + str(self.end), tracer_data)
        np.save(self.directory_data + 'Velocity_data_' + str(self.start) + '_to_' + str(self.end), vel_data)

    def PCA(self, field_name):

        field_name = field_name
        start_interv = self.start_pca
        end_interv = self.end_pca
        observationPeriod = 'data_' + str(start_interv) + '_to_' + str(end_interv)
        modelData = np.load(self.directory_data + observationPeriod + '/' + field_name + '_' + observationPeriod + '.npy')

        if 'Velocity' in field_name:
            modelData = np.reshape(modelData, (modelData.shape[0], modelData.shape[1] * modelData.shape[2]), order='F')

        # Standardise the data with mean 0
        meanData = np.nanmean(modelData, 0)
        stdData = np.nanstd(modelData)
        modelDataScaled = (modelData - meanData) / stdData

        #PCA solver
        solver = Eof(modelDataScaled)

        #Principal Components time-series
        pcs = solver.pcs()
        #Projection
        eof = solver.eofs()


        np.save(self.directory_data + '/' + 'pcs_' + field_name + '_' + observationPeriod,
                pcs)
        np.save(self.directory_data + '/' + 'eofs_' + field_name + '_' + observationPeriod,
                eof)
        np.save(self.directory_data + '/' + 'varCumulative_' + field_name + '_' + observationPeriod,
                varianceCumulative)
        np.save(self.directory_data + '/' + 'eigenvalues_' + field_name + '_' + observationPeriod,
                eigenvalues)
        np.save(self.directory_data + '/' + 'mean_' + field_name + '_' + observationPeriod,
                meanData)
        np.save(self.directory_data + '/' + 'std_' + field_name + '_' + observationPeriod,
                stdData)
