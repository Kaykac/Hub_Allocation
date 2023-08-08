"""
The "Cluster" class in this code is derived from the "Ui_MainWindow" class.
This code is a piece of code that contains the working principles of K_Means, Affinity Propagation, MeanShift, Spectral and Hierarchical Clustering algorithms.
In addition, KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering and DBSCAN libraries were used.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QListWidgetItem
from Oop_Proje import Ui_MainWindow

from PyQt5.QtWidgets import  QMainWindow, QLabel, QVBoxLayout, QWidget, QSpinBox, QPushButton,QHBoxLayout,QComboBox,QDoubleSpinBox

class Cluster(Ui_MainWindow):
    def __init__(self, MainWindow):
        
        

        
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(MainWindow)
        
        self.ui.toolButton_C1.clicked.connect(self.K_MeansWindow)
        self.ui.action_K_Means.triggered.connect(self.K_MeansWindow)
        
        self.ui.toolButton_C2.clicked.connect(self.AffinityWindow)
        self.ui.action_Affinitiy_Propagation.triggered.connect(self.AffinityWindow)
        
        self.ui.toolButton_C3.clicked.connect(self.MeanShift)
        self.ui.action_Means_Shift.triggered.connect(self.MeanShift)
        
        self.ui.toolButton_C4.clicked.connect(self.SpectralWindow)
        self.ui.action_Spectral_Clustering.triggered.connect(self.SpectralWindow)
        
        self.ui.toolButton_C5.clicked.connect(self.HierarchicalWindow)
        self.ui.action_Hierarchical_Clustering.triggered.connect(self.HierarchicalWindow)
        
        self.ui.toolButton_C6.clicked.connect(self.DBSCAN_Window)
        self.ui.action_DBSCAN.triggered.connect(self.DBSCAN_Window)

    # # @brief Enable the necessary UI elements for the EnebCluster operation.
    #   @details Enables the specified tool buttons and actions in the user interface to perform the EnebCluster operation.
    #   @return None    
    def EnebCluster(self):
        self.ui.toolButton_I1.setEnabled(True)
        self.ui.toolButton_I2.setEnabled(True)
        self.ui.toolButton_I3.setEnabled(True)
        self.ui.toolButton_I4.setEnabled(True)
        self.ui.toolButton_I5.setEnabled(True)
        self.ui.action_Save_Initial_Solution.setEnabled(True)
        self.ui.action_Initial_Solution.setEnabled(True)
        self.ui.action_Initial_Solution_2.setEnabled(True)
        self.ui.action_Initial_Solution_3.setEnabled(True)
        self.ui.action_Initial_Solution_4.setEnabled(True)
        
        
    ## @brief Opens the K-Means window.
    #  @details Opens a window that allows the user to select the parameters of the K-Means algorithm.
    #  @return None
    def K_MeansWindow(self):
        self.new_window = QMainWindow()
        self.new_window.setWindowTitle("K-Means")
    
        labels = ["n_clusters:", "n_init:", "max_iter:"]
        
        max_values = [50, 100, 1000]
        
        default_values = [3, 10, 100]
        
        spinbox1 = QSpinBox()
        spinbox1.setRange(1, max_values[0])
        spinbox1.setValue(default_values[0])
        
        spinbox2 = QSpinBox()
        spinbox2.setRange(1, max_values[1])
        spinbox2.setValue(default_values[1])
        
        spinbox3 = QSpinBox()
        spinbox3.setRange(1, max_values[2])
        spinbox3.setValue(default_values[2])
        
        button_ok = QPushButton("OK")
        button_ok.clicked.connect(lambda: self.KMeans(spinbox1.value(), spinbox2.value(), spinbox3.value()))
        
        layout = QVBoxLayout()
        label1 = QLabel(labels[0])
        label2 = QLabel(labels[1])
        label3 = QLabel(labels[2])
        
        hbox1 = QHBoxLayout()
        hbox1.addWidget(label1)
        hbox1.addWidget(spinbox1)
        
        hbox2 = QHBoxLayout()
        
        hbox2.addWidget(label2)
        hbox2.addWidget(spinbox2)
        
        hbox3 = QHBoxLayout()
        hbox3.addWidget(label3)
        hbox3.addWidget(spinbox3)
        
        layout.addLayout(hbox1)
        layout.addLayout(hbox2)
        layout.addLayout(hbox3)

        
        layout.addStretch()
        layout.addWidget(button_ok)
        
        widget = QWidget()
        widget.setLayout(layout)
        self.new_window.setCentralWidget(widget)
        self.new_window.setFixedSize(200, 150)
        self.new_window.show()
    
    ## @brief Opens the Hierarchical window.
    #  @details Opens a window that allows the user to select the parameters of the Hierarchical algorithm.
    #  @return None   
    def HierarchicalWindow(self):
        self.new_window = QMainWindow()
        self.new_window.setWindowTitle("Hierarchical")
        
        labels = ["n_clusters:", "affinity:", "linkage:"]
        
        spinbox_n_clusters = QSpinBox()
        spinbox_n_clusters.setRange(1, 50)
        spinbox_n_clusters.setValue(3)
        
        combobox_affinity = QComboBox()
        combobox_affinity.addItem("euclidean")
        combobox_affinity.addItem("l1")
        combobox_affinity.addItem("l2")
        combobox_affinity.addItem("manhattan")
        combobox_affinity.addItem("cosine")
   
        combobox_linkage = QComboBox()
        combobox_linkage.addItem("ward")
        combobox_linkage.addItem("complete")
        combobox_linkage.addItem("average")
        combobox_linkage.addItem("single")
        
        button_ok = QPushButton("OK")
        button_ok.clicked.connect(lambda: self.Hierarchical(spinbox_n_clusters.value(), combobox_affinity.currentText(), combobox_linkage.currentText()))
        
        layout = QVBoxLayout()
        
        label_cluster_name1 = QLabel(labels[0])
        hbox1 = QHBoxLayout()
        hbox1.addWidget(label_cluster_name1)
        hbox1.addWidget(spinbox_n_clusters)
        layout.addLayout(hbox1)
        
        label_cluster_name2 = QLabel(labels[1])
        hbox2 = QHBoxLayout()
        hbox2.addWidget(label_cluster_name2)
        hbox2.addWidget(combobox_affinity)
        layout.addLayout(hbox2)
        
        label_cluster_name3 = QLabel(labels[2])
        hbox3 = QHBoxLayout()
        hbox3.addWidget(label_cluster_name3)
        hbox3.addWidget(combobox_linkage)
        layout.addLayout(hbox3)
        
        layout.addStretch()
        layout.addWidget(button_ok)
        
        widget = QWidget()
        widget.setLayout(layout)
        self.new_window.setCentralWidget(widget)
        self.new_window.setFixedSize(200, 150)
        self.new_window.show()

    ## @brief Opens the Spectral window.
    #  @details Opens a window that allows the user to select the parameters of the Spectral algorithm.
    #  @return None    
    def SpectralWindow(self):
         self.new_window = QMainWindow()
         self.new_window.setWindowTitle("Spectral")
         
         labels = ["n_clusters:", "n_init:"]
         
         max_values = [50, 100]
         
         default_values = [3, 10]
         
         spinbox1 = QSpinBox()
         spinbox1.setRange(1, max_values[0])
         spinbox1.setValue(default_values[0])
        
         spinbox2 = QSpinBox()
         spinbox2.setRange(1, max_values[1])
         spinbox2.setValue(default_values[1])
        
         spinboxes = [spinbox1, spinbox2]
          
         button_ok = QPushButton("OK")
         button_ok.clicked.connect(lambda: self.Spectral(*[spinbox.value() for spinbox in spinboxes]))
         
         layout = QVBoxLayout()
         for label, spinbox in zip(labels, spinboxes):
             label_cluster_name = QLabel(label)           
             hbox = QHBoxLayout()
             hbox.addWidget(label_cluster_name)
             hbox.addWidget(spinbox)
             layout.addLayout(hbox)
         
         layout.addStretch()
         layout.addWidget(button_ok)
         
         widget = QWidget()
         widget.setLayout(layout)
         self.new_window.setCentralWidget(widget)
         self.new_window.setFixedSize(200, 100)
         self.new_window.show()
         
         
         
    ## @brief Opens the Affinity window.
    #  @details Opens a window that allows the user to select the parameters of the Affinity algorithm.
    #  @return None    
    def AffinityWindow(self):
        self.new_window = QMainWindow()
        self.new_window.setWindowTitle("Affinity")

        labels = ["damping:", "max_init:", "convergence_iter:"]

        max_values = [0.99, 1000, 100]
        min_values = [0.5, 1, 1]

        default_values = [0.5, 200, 15]

        double_spinbox = QDoubleSpinBox()
        double_spinbox.setRange(min_values[0], max_values[0])
        double_spinbox.setSingleStep(0.01)
        double_spinbox.setValue(default_values[0])

        spinboxes = [QSpinBox() for _ in range(2)]
        for i, spinbox in enumerate(spinboxes):
            spinbox.setRange(min_values[0], max_values[i+1])
            spinbox.setValue(default_values[i+1])

        button_ok = QPushButton("OK")
        button_ok.clicked.connect(lambda: self.Affinity(double_spinbox.value(), *[spinbox.value() for spinbox in spinboxes]))

        layout = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        
        label_cluster1 = QLabel(labels[0])
        label_cluster2 = QLabel(labels[1])
        label_cluster3 = QLabel(labels[2])
        
        hbox1.addWidget(label_cluster1)
        hbox1.addWidget(double_spinbox)
        
        hbox2.addWidget(label_cluster2)
        hbox2.addWidget(spinboxes[0])
        
        hbox3.addWidget(label_cluster3)
        hbox3.addWidget(spinboxes[1])
        
        layout.addLayout(hbox1)
        layout.addLayout(hbox2)
        layout.addLayout(hbox3)
        layout.addWidget(button_ok)

        widget = QWidget()
        widget.setLayout(layout)
        self.new_window.setCentralWidget(widget)
        self.new_window.setFixedSize(200, 150)
        self.new_window.show()
        
        
        
        
         
    ## @brief Opens the DBSCAN window.
    #  @details Opens a window that allows the user to select the parameters of the DBSCAN algorithm.
    #  @return None         
    def DBSCAN_Window(self):
        self.new_window = QMainWindow()
        self.new_window.setWindowTitle("DBSCAN")
        
        labels = ["eps:", "min_samples:"]

        max_values = [10000, 50]
        
        default_values = [5000, 2]
        
        spinboxes = [QSpinBox() for _ in range(2)]
        for i, spinbox in enumerate(spinboxes):
            spinbox.setRange(1, max_values[i])
            spinbox.setValue(default_values[i])
        
        button_ok = QPushButton("OK")
        button_ok.clicked.connect(lambda: self.DBSCAN(*[spinbox.value() for spinbox in spinboxes]))
        
        layout = QVBoxLayout()
        for label, spinbox in zip(labels, spinboxes):
            label_cluster_name = QLabel(label)
            
            hbox = QHBoxLayout()
            hbox.addWidget(label_cluster_name)
            hbox.addWidget(spinbox)
            layout.addLayout(hbox)
        
        layout.addStretch()
        layout.addWidget(button_ok)
        
        widget = QWidget()
        widget.setLayout(layout)
        self.new_window.setCentralWidget(widget)
        self.new_window.setFixedSize(200, 100)
        self.new_window.show()
         
         
         
 
               
               
    ## @brief runs the K-Means algorithm.
    #  @details Runs the K-Means algorithm with the parameters selected by the user and displays the results.
    #  @param n_clusters Number of clusters
    #  @param n_init Number of starting points
    #  @param max_iter Maximum number of iterations
    #  @return None 
    def KMeans(self, n_clusters, n_init, max_iter):
        
        plt.clf()
        data = np.loadtxt('Points.txt')
    
        model = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)  
        model.fit(data)
    
        self.cluster_centers = model.cluster_centers_
        labels = model.labels_
    
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        plt.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], marker='*', color='red', s=100, label='centroids')
        plt.legend()
        plt.title('K-Means Clustering')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.savefig("Plot.jpg", format='jpeg')
    
        pixmap = QPixmap("Plot.jpg")
        self.ui.label_In.setPixmap(pixmap)
        self.ui.label_In.adjustSize()
        self.EnebCluster()
    
        self.ui.listWidget_Info.clear()
        item_text = "(K-Means Clustering)"
        list_item = QListWidgetItem(item_text)
        self.ui.listWidget_Info.addItem(list_item)
    
        self.centers = self.cluster_centers
        for i in range(len(self.cluster_centers)):
            x = self.cluster_centers[i][0]
            y = self.cluster_centers[i][1]
            item_text = f"\nCentroid of Cluster {i+1}: ({x:.2f}, {y:.2f})"
            list_item = QListWidgetItem(item_text)
            self.ui.listWidget_Info.addItem(list_item)
    
            cluster_points = data[labels == i]
            for point in cluster_points:
                x = point[0]
                y = point[1]
                item_text = f"Point: ({x:.2f}, {y:.2f})"
                list_item = QListWidgetItem(item_text)
                self.ui.listWidget_Info.addItem(list_item)
    
        self.ui.toolButton_H1.setEnabled(True)
        self.ui.toolButton_H2.setEnabled(True)
        self.ui.action_Simulated_Anneling.setEnabled(True)
        self.ui.action_Hill_Climbing.setEnabled(True)
        self.new_window.close()
        print(self.cluster_centers)
        
                      
              
        
        
        
   
    ## @brief Runs the Hierarchical clustering algorithm.
    #  @details Runs the Hierarchical clustering algorithm with the parameters selected by the user and displays the results.
    #  @param n_clusters Number of clusters
    #  @param affinity Similarity measure
    #  @param linkage Merge method
    #  @return None
    def Hierarchical(self, n_clusters, affinity, linkage):
        plt.clf()
        data = np.loadtxt('Points.txt')
    
        model = AgglomerativeClustering(n_clusters=n_clusters,affinity=affinity,linkage=linkage)  
        labels = model.fit_predict(data)
    
        unique_labels = np.unique(labels)
        self.cluster_centers = []
        for label in unique_labels:
            cluster_points = data[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            self.cluster_centers.append(cluster_center)
    
        self.cluster_centers = np.array(self.cluster_centers)
    
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        plt.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], marker='*', color='red', s=100, label='centroids')
        plt.legend()
        plt.title('Hierarchical Clustering')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.savefig("Plot.jpg", format='jpeg')
    
        pixmap = QPixmap("Plot.jpg")
        self.ui.label_In.setPixmap(pixmap)
        self.ui.label_In.adjustSize()
        self.EnebCluster()
    
        self.ui.listWidget_Info.clear()
        item_text = "(Hierarchical Clustering)"
        list_item = QListWidgetItem(item_text)
        self.ui.listWidget_Info.addItem(list_item)
    
        for i in range(len(self.cluster_centers)):
            x = self.cluster_centers[i][0]
            y = self.cluster_centers[i][1]
            item_text = f"\nCentroid of Cluster {i+1}: ({x:.2f}, {y:.2f})"
            list_item = QListWidgetItem(item_text)
            self.ui.listWidget_Info.addItem(list_item)
    
            cluster_points = data[labels == i]
            for point in cluster_points:
                x = point[0]
                y = point[1]
                item_text = f"Point: ({x:.2f}, {y:.2f})"
                list_item = QListWidgetItem(item_text)
                self.ui.listWidget_Info.addItem(list_item)
    
        self.ui.toolButton_H1.setEnabled(True)
        self.ui.toolButton_H2.setEnabled(True)
        self.ui.action_Simulated_Anneling.setEnabled(True)
        self.ui.action_Hill_Climbing.setEnabled(True)
        self.new_window.close()

 
    ## @brief Runs the Spectral clustering algorithm.
    #  @details Runs the Spectral clustering algorithm with the parameters selected by the user and displays the results.
    #  @param n_clusters Number of clusters
    #  @param n_init Start number
    #  @return None    
    def Spectral(self, n_clusters, n_init):
        plt.clf()
        data = np.loadtxt('Points.txt')
    
        model = SpectralClustering(n_clusters=n_clusters,n_init=n_init)  
        labels = model.fit_predict(data)
    
        unique_labels = np.unique(labels)
        self.cluster_centers = []
        for label in unique_labels:
            cluster_points = data[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            self.cluster_centers.append(cluster_center)
    
        self.cluster_centers = np.array(self.cluster_centers)
    
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        plt.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], marker='*', color='red', s=100, label='centroids')
        plt.legend()
        plt.title('Spectral Clustering')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.savefig("Plot.jpg", format='jpeg')
    
        pixmap = QPixmap("Plot.jpg")
        self.ui.label_In.setPixmap(pixmap)
        self.ui.label_In.adjustSize()
        self.EnebCluster()
    
        self.ui.listWidget_Info.clear()
        item_text = "(Spectral Clustering)"
        list_item = QListWidgetItem(item_text)
        self.ui.listWidget_Info.addItem(list_item)
    
        for i in range(len(self.cluster_centers)):
            x = self.cluster_centers[i][0]
            y = self.cluster_centers[i][1]
            item_text = f"\nCentroid of Cluster {i+1}: ({x:.2f}, {y:.2f})"
            list_item = QListWidgetItem(item_text)
            self.ui.listWidget_Info.addItem(list_item)
    
            cluster_points = data[labels == i]
            for point in cluster_points:
                x = point[0]
                y = point[1]
                item_text = f"Point: ({x:.2f}, {y:.2f})"
                list_item = QListWidgetItem(item_text)
                self.ui.listWidget_Info.addItem(list_item)
    
        self.ui.toolButton_H1.setEnabled(True)
        self.ui.toolButton_H2.setEnabled(True)
        self.ui.action_Simulated_Anneling.setEnabled(True)
        self.ui.action_Hill_Climbing.setEnabled(True)
        self.new_window.close()
        
    ## @brief Runs the Mean Shift clustering algorithm.
    #  @details Runs the Mean Shift clustering algorithm with the given bandwidth value and displays the results.
    #  @param bandwidth Bandwidth value
    #  @return None      
    def MeanShift(self,bandwidth):
    
        plt.clf()
        data = np.loadtxt('Points.txt')
    
        model = MeanShift()
        model.fit(data)
    
        self.cluster_centers = model.cluster_centers_
        labels = model.labels_
    
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        plt.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], marker='*', color='red', s=100, label='centroids')
        plt.legend()
        plt.title('Mean Shift Clustering')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.savefig("Plot.jpg", format='jpeg')
    
        pixmap = QPixmap("Plot.jpg")
        self.ui.label_In.setPixmap(pixmap)
        self.ui.label_In.adjustSize()
        self.EnebCluster()
    
        self.ui.listWidget_Info.clear()
        item_text = "(Mean Shift Clustering)"
        list_item = QListWidgetItem(item_text)
        self.ui.listWidget_Info.addItem(list_item)
    
        self.centers = self.cluster_centers
        for i in range(len(self.cluster_centers)):
            x = self.cluster_centers[i][0]
            y = self.cluster_centers[i][1]
            item_text = f"\nCentroid of Cluster {i+1}: ({x:.2f}, {y:.2f})"
            list_item = QListWidgetItem(item_text)
            self.ui.listWidget_Info.addItem(list_item)
    
            cluster_points = data[labels == i]
            for point in cluster_points:
                x = point[0]
                y = point[1]
                item_text = f"Point: ({x:.2f}, {y:.2f})"
                list_item = QListWidgetItem(item_text)
                self.ui.listWidget_Info.addItem(list_item)
    
        self.ui.toolButton_H1.setEnabled(True)
        self.ui.toolButton_H2.setEnabled(True)
        self.ui.action_Simulated_Anneling.setEnabled(True)
        self.ui.action_Hill_Climbing.setEnabled(True)
        #print(len(self.cluster_centers))
        
        
        
            
    
    ## @brief Runs the Affinity Propagation clustering algorithm.
    #  @details Runs the Affinity Propagation clustering algorithm with the parameters selected by the user and displays the results.
    #  @param dumping dumping factor
    #  @param max_iter Maximum number of iterations
    #  @param convergence_iter Number of iterations of convergence
    #  @return None    
    def Affinity(self,damping,max_iter,convergence_iter):
        plt.clf()
        data = np.loadtxt('Points.txt')
    
        model = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter)
        model.fit(data)
    
        self.cluster_centers = model.cluster_centers_
        labels = model.labels_
    
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        plt.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], marker='*', color='red', s=100, label='centroids')
        plt.legend()
        plt.title('Affinity Propagation Clustering')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.savefig("Plot.jpg", format='jpeg')
    
        pixmap = QPixmap("Plot.jpg")
        self.ui.label_In.setPixmap(pixmap)
        self.ui.label_In.adjustSize()
        self.EnebCluster()
    
        self.ui.listWidget_Info.clear()
        item_text = "(Affinity Propagation Clustering)"
        list_item = QListWidgetItem(item_text)
        self.ui.listWidget_Info.addItem(list_item)
    
        self.centers = self.cluster_centers
        for i in range(len(self.cluster_centers)):
            x = self.cluster_centers[i][0]
            y = self.cluster_centers[i][1]
            item_text = f"\nCentroid of Cluster {i+1}: ({x:.2f}, {y:.2f})"
            list_item = QListWidgetItem(item_text)
            self.ui.listWidget_Info.addItem(list_item)
    
            cluster_points = data[labels == i]
            for point in cluster_points:
                x = point[0]
                y = point[1]
                item_text = f"Point: ({x:.2f}, {y:.2f})"
                list_item = QListWidgetItem(item_text)
                self.ui.listWidget_Info.addItem(list_item)
    
        self.ui.toolButton_H1.setEnabled(True)
        self.ui.toolButton_H2.setEnabled(True)
        self.ui.action_Simulated_Anneling.setEnabled(True)
        self.ui.action_Hill_Climbing.setEnabled(True)
        self.new_window.close()
                
                
                
    ## @brief Runs DBSCAN clustering algorithm.
    #  @details Runs the DBSCAN clustering algorithm with the parameters selected by the user and displays the results.
    #  @param eps Epsilon value
    #  @param min_samples Minimum number of samples
    #  @return None        
    def DBSCAN(self,eps,min_samples):
        plt.clf()
        data = np.loadtxt('Points.txt')
    
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(data)
    
        labels = model.labels_
        unique_labels = np.unique(labels)
    
        self.cluster_centers = []
        for label in unique_labels:
            if label != -1:
                cluster_points = data[labels == label]
                cluster_center = np.mean(cluster_points, axis=0)
                self.cluster_centers.append(cluster_center)
    
        self.cluster_centers = np.array(self.cluster_centers)
    
        outliers = data[model.labels_ == -1]
    
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        plt.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], marker='*', color='red', s=100, label='Centroids')
        plt.scatter(outliers[:, 0], outliers[:, 1], color='black', marker='x', label='Outlier Points', s=200)
        plt.title('DBSCAN')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        plt.savefig("Plot.jpg", format='jpeg')
    
        pixmap = QPixmap("Plot.jpg")
        self.ui.label_In.setPixmap(pixmap)
        self.ui.label_In.adjustSize()
        self.EnebCluster()
    
        self.ui.listWidget_Info.clear()
        item_text = "(DBSCAN)"
        list_item = QListWidgetItem(item_text)
        self.ui.listWidget_Info.addItem(list_item)
    
        for i, label in enumerate(unique_labels):
            if label != -1:
                cluster_points = data[labels == label]
                item_text = f"Cluster {i}: Points: {len(cluster_points)}"
                list_item = QListWidgetItem(item_text)
                self.ui.listWidget_Info.addItem(list_item)
    
                cluster_center = self.cluster_centers[i - 1]
                x = cluster_center[0]
                y = cluster_center[1]
                center_text = f"Cluster {i} Center: ({x:.2f}, {y:.2f})"
                center_list_item = QListWidgetItem(center_text)
                self.ui.listWidget_Info.addItem(center_list_item)
        if len(outliers) > 0:
            item_text = f"Outliers: {len(outliers)}"
            list_item = QListWidgetItem(item_text)
            self.ui.listWidget_Info.addItem(list_item)
    
        self.ui.toolButton_H1.setEnabled(True)
        self.ui.toolButton_H2.setEnabled(True)
        self.ui.action_Simulated_Anneling.setEnabled(True)
        self.ui.action_Hill_Climbing.setEnabled(True)
        self.new_window.close()
            
    
          
