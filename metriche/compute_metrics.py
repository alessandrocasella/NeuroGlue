import ssim_processing as sp
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import util_fun
import pandas as pd

import argparse

colors = ['orange', 'blue', 'red', 'green', 'yellow']
if __name__ == '__main__':
	videoToTest = "/home/amdeluca/metriche/data2" 
	
	print(videoToTest)
 
	experiments = ["/brisk", "/orb", "/SG", "/SG_trained", "/sift"]
	#experiments = ['output_sift_RANSAC', 'baseline', 'output_sp_RANSAC']
	AllExp = []
	SVDExp = []
	for experiment in experiments:
		print(experiment)
		if os.path.isdir(os.path.join(videoToTest + experiment)) and experiment != 'images':
			H_list = [np.loadtxt(f) for f in sorted(glob.glob(os.path.join(videoToTest + experiment + "/hom/" + "*.txt")))]
			#print(os.path.join(videoToTest + experiment + '/boxplot.png'))
			frame_list = [os.path.join(videoToTest  + experiment + "/images/mosaic" + "{}.jpg".format(os.path.basename(f)[3:-4])) for f in sorted(glob.glob(os.path.join(videoToTest + experiment + "/hom/" + "*.txt")))]
			#print(os.path.join(videoToTest + '/metrics.svg'))
			ssim_matrix = sp.getSSIMForFrameDistance(1, 5, H_list, 'Homography', 256, frame_list, False)
			print(ssim_matrix.shape)
			df = pd.DataFrame(ssim_matrix)
			path = os.path.join(videoToTest + experiment + "/ssim_matrix.xlsx")
			df.to_excel(excel_writer = path)
			print('Median: {:.4f}'.format(np.median(ssim_matrix[:, 4])))
			print('Mean {:.4f} +/- {:.4f}:'.format(np.mean(ssim_matrix[:, 4]), np.std(ssim_matrix[:, ])))
			AllExp.append(ssim_matrix[:, 4])
			SVDExp.append(util_fun.SVD(H_list))
			fig = plt.figure(figsize=(10, 7))

			# Creating axes instance
			ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])

			# Creating plot
			bp = ax.boxplot(ssim_matrix)

			ax.set_title("box plot {} {}".format(experiment, videoToTest))
			ax.set_xlabel("frame Index")
			ax.set_ylabel("SSIM")
			plt.ylim([0, 1])
			# show plot
			plt.savefig(os.path.join(videoToTest + experiment + '/boxplot_2.png'))
			plt.show()
	# trend param
	fig, axs = plt.subplots(7, 1, figsize=(7, 14))
	fig.suptitle("Metrics {}".format(videoToTest))
	for a, el in enumerate(AllExp):
		axs[0].set_title('SSIM')
		axs[0].set_ylim([0, 1])
		axs[0].plot(el, color=colors[a])

	params = ['sx', 'sy', 'theta', 'gamma', 'tx', 'ty']
	#plt.plot(AllExp[a], color=colors[0], label='SSIM')

	for a, el in enumerate(SVDExp):
		for b in range(len(params)):
			axs[b + 1].set_title(params[b])
			axs[b + 1].plot(el[b], color=colors[a])
		plt.legend(loc="upper left")
	plt.savefig(os.path.join(videoToTest + '/metrics_2.svg'))
	plt.savefig(os.path.join(videoToTest + '/metrics_2.png'))
	plt.show()