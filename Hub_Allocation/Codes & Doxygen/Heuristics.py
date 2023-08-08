"""
The "Heuristics" class in this code is derived from the "Cluster" class. 
This code is a piece of code that contains the working principles of HillClimb and Simulated Annealing algorithms.
"""
from Cluster import Cluster

import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap
from numpy.random import randn
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QSpinBox, QPushButton,QHBoxLayout,QComboBox,QDoubleSpinBox,QSpacerItem, QSizePolicy, QListWidgetItem


class Heuristics(Cluster):
    def __init__(self, MainWindow):
        super().__init__(MainWindow)
        
        self.ui.action_Hill_Climbing.triggered.connect(self.HillClimbWindow)
        self.ui.toolButton_H1.clicked.connect(self.HillClimbWindow)
        self.ui.action_Simulated_Anneling.triggered.connect(self.AnnealingWindow)
        self.ui.toolButton_H2.clicked.connect(self.AnnealingWindow)
        
         
    ## @brief Opens the HillClimb window.
    #  @details Opens a window that allows the user to select the parameters of the HillClimb algorithm.
    #  @return None
    def HillClimbWindow(self):
        self.new_window = QMainWindow()
        self.new_window.setWindowTitle("HillClimb")

        labels = ["n_iterations:", "step_size:"]

        max_values = [1.0, 10000]
        min_values = [0.01, 1000]

        double_spinbox = QDoubleSpinBox()
        double_spinbox.setRange(min_values[0], max_values[0])
        double_spinbox.setSingleStep(0.01)
        double_spinbox.setValue(0.5)

        spinbox=QSpinBox()
        spinbox.setRange(0, 10000)
        spinbox.setSingleStep(50)
        spinbox.setValue(1000)

        button_ok = QPushButton("OK")
        button_ok.clicked.connect(lambda: self.hill_climbing(spinbox.value(),double_spinbox.value()))

        layout = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        
        label_cluster1 = QLabel(labels[0])
        label_cluster2 = QLabel(labels[1])
        
        hbox1.addWidget(label_cluster1)
        hbox1.addWidget(double_spinbox)
        
        hbox2.addWidget(label_cluster2)
        hbox2.addWidget(spinbox)

        layout.addLayout(hbox1)
        layout.addLayout(hbox2)
        
        layout.addWidget(button_ok)

        widget = QWidget()
        widget.setLayout(layout)
        self.new_window.setCentralWidget(widget)
        self.new_window.setFixedSize(200, 150)
        self.new_window.show()        

    ## @brief Runs the Hill Climbing algorithm.
    #  @details Runs the Hill Climbing algorithm with the given n_iterations and step_size values and displays the results.
    #  @param n_iterations Number of iterations
    #  @param step_size Step size
    #  @return None
    def hill_climbing(self, n_iterations,step_size):
        # # @brief Calculate the objective function value for a given set of cluster centers and data points.
        # @param cluster_centers The coordinates of the cluster centers.
        # @param data The data points.
        # @return The total distance.             
        def objective(cluster_centers, data):
            total_distance = 0
            for point in data:
                distances = [np.linalg.norm(point - center) for center in cluster_centers]
                total_distance += min(distances)
            return total_distance  
        

        self.new_window.close()
          

        data = np.loadtxt('Points.txt')
        cluster_centers  = self.cluster_centers
        n_iterations = n_iterations
        step_size=step_size
        
        current_solution = cluster_centers.copy()
        best_solution = current_solution.copy()
        
        for _ in range(n_iterations):
            candidate = current_solution + randn(*current_solution.shape) * step_size
            if objective(candidate, data) < objective(current_solution, data):
                current_solution = candidate.copy()
                if objective(candidate, data) < objective(best_solution, data):
                    best_solution = candidate.copy()

        print(best_solution)

        num_clusters = len(best_solution)
        color_palette = plt.cm.get_cmap('tab10', num_clusters)

        for i in range(num_clusters):
            cluster_points = data[np.where(np.argmin(np.linalg.norm(data - best_solution[i], axis=1)) == i)]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color_palette(i), label=f'Cluster {i+1}')

        plt.scatter(best_solution[:, 0], best_solution[:, 1], color='red', marker='*', label='Centroids1',s=150,edgecolor='black')
        plt.title('Hill Climbing')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.savefig("Plot1.jpg", format='jpeg')
        
        

        pixmap = QPixmap("Plot1.jpg")
        self.ui.label__Fin.setPixmap(pixmap)
        self.ui.label__Fin.adjustSize()

        self.ui.toolButton_F1.setEnabled(True)
        self.ui.toolButton_F2.setEnabled(True)
        self.ui.toolButton_F3.setEnabled(True)
        self.ui.toolButton_F4.setEnabled(True)
        self.ui.toolButton_F5.setEnabled(True)

        self.ui.action_Save_Final_Solution.setEnabled(True)
        self.ui.action_Final_Solution.setEnabled(True)
        self.ui.action_Final_Solution_2.setEnabled(True)
        self.ui.action_Final_Solution_3.setEnabled(True)
        self.ui.action_Final_Solution_4.setEnabled(True)
        
        
        
        
        
        self.ui.listWidget_Result.clear()
        item_text = "(Best solutions after The Hill Climbing:)\n"
        list_item = QListWidgetItem(item_text)
        self.ui.listWidget_Result.addItem(list_item)
    
        for i in range(len(best_solution)):
            x = best_solution[i][0]
            y = best_solution[i][1]
            item_text = f"Centroid of Cluster {i+1}: ({x:.2f}, {y:.2f})"
            list_item = QListWidgetItem(item_text)
            self.ui.listWidget_Result.addItem(list_item)
            
            
    ## @brief Opens the Simulated Annealing window.
    #  @details Opens a window that allows the user to select the parameters of the Simulated Annealing algorithm.
    #  @return None                
    def AnnealingWindow(self):
        self.new_window = QMainWindow()
        self.new_window.setWindowTitle("Annealing")

        labels = ["n_iterations:", "step_size:", "initial_temp:"]

        max_values = [10000, 1.0, 100.0]
        min_values = [1000, 0.01, 0.1]

        default_values = [1000, 0.5, 10.0]

        spin_box1 = QSpinBox()
        spin_box1.setRange(min_values[0], max_values[0])
        spin_box1.setSingleStep(50)
        spin_box1.setValue(default_values[0])
        
        double_spin_box1 = QDoubleSpinBox()
        double_spin_box1.setRange(min_values[1], max_values[1])
        double_spin_box1.setSingleStep(0.01)
        double_spin_box1.setValue(default_values[1])
        
        double_spin_box2 = QDoubleSpinBox()
        double_spin_box2.setRange(min_values[2], max_values[2])
        double_spin_box2.setSingleStep(0.1)
        double_spin_box2.setValue(default_values[2])

        button_ok = QPushButton("OK")
        button_ok.clicked.connect(lambda: self.annealing(n_iterations=spin_box1.value(), step_size=double_spin_box1.value(), initial_temp=double_spin_box2.value()))

        layout = QVBoxLayout()
        
        hbox = QHBoxLayout()
        label = QLabel(labels[0])
        hbox.addWidget(label)
        hbox.addWidget(spin_box1)
        layout.addLayout(hbox)
        
        hbox = QHBoxLayout()
        label = QLabel(labels[1])
        hbox.addWidget(label)
        hbox.addWidget(double_spin_box1)
        layout.addLayout(hbox)
        
        hbox = QHBoxLayout()
        label = QLabel(labels[2])
        hbox.addWidget(label)
        hbox.addWidget(double_spin_box2)
        layout.addLayout(hbox)

        
        layout.addWidget(button_ok)

        widget = QWidget()
        widget.setLayout(layout)
        self.new_window.setCentralWidget(widget)
        self.new_window.setFixedSize(200, 150)
        self.new_window.show()

    
    ## @brief Runs the Simulated Annealing algorithm.
    #  @details Runs the Simulated Annealing algorithm with the given n_iterations, step_size and initial_temp values and displays the results.
    #  @param n_iterations Number of iterations
    #  @param step_size Step size
    #  @param initial_temp Initial temperature
    #  @return None                        
    def annealing(self, n_iterations, step_size, initial_temp):
        
        # # @brief Calculate the objective function value for a given set of cluster centers and data points.
        # @param cluster_centers The coordinates of the cluster centers.
        # @param data The data points.
        # @return The total distance.        
        def objective(cluster_centers, data):
            total_distance = 0
            for point in data:
                distances = [np.linalg.norm(point - center) for center in cluster_centers]
                total_distance += min(distances)
            return total_distance
    
        self.new_window.close()
    
        data = np.loadtxt('Points.txt')
        cluster_centers = self.cluster_centers.copy()
        best_solution = cluster_centers.copy()
    
        curr_solution = cluster_centers.copy()
        curr_eval = objective(curr_solution, data)
        best_eval = curr_eval
    
        temperature = initial_temp
    
        for i in range(n_iterations):
            candidate_solution = curr_solution + np.random.randn(*curr_solution.shape) * step_size
            candidate_eval = objective(candidate_solution, data)
    
            diff = candidate_eval - curr_eval
            metropolis = np.exp(-diff / temperature)
    
            if diff < 0 or np.random.rand() < metropolis:
                curr_solution = candidate_solution
                curr_eval = candidate_eval
    
            if candidate_eval < best_eval:
                best_solution = candidate_solution
                best_eval = candidate_eval
    
            temperature = 1 / (i + 1)
    
        print(best_solution)
    
        num_clusters = len(best_solution)
        color_palette = plt.cm.get_cmap('tab10', num_clusters)
    
        for i in range(num_clusters):
            cluster_points = data[np.where(np.argmin(np.linalg.norm(data - best_solution[i], axis=1)) == i)]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color_palette(i), label=f'Cluster {i+1}')
    
        plt.scatter(best_solution[:, 0], best_solution[:, 1], color='red', marker='*', label='Centroids1', s=150, edgecolor='black')
        plt.title('Simulated Annealing')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.savefig("Plot1.jpg", format='jpeg')
    
        pixmap = QPixmap("Plot1.jpg")
        self.ui.label__Fin.setPixmap(pixmap)
        self.ui.label__Fin.adjustSize()
    
        self.ui.toolButton_F1.setEnabled(True)
        self.ui.toolButton_F2.setEnabled(True)
        self.ui.toolButton_F3.setEnabled(True)
        self.ui.toolButton_F4.setEnabled(True)
        self.ui.toolButton_F5.setEnabled(True)
    
        self.ui.action_Save_Final_Solution.setEnabled(True)
        self.ui.action_Final_Solution.setEnabled(True)
    
        self.ui.listWidget_Result.clear()
        item_text = "(Best solutions after Simulated Annealing:)\n"
        list_item = QListWidgetItem(item_text)
        self.ui.listWidget_Result.addItem(list_item)
    
        for i in range(len(best_solution)):
            x = best_solution[i][0]
            y = best_solution[i][1]
            item_text = f"Centroid of Cluster {i+1}: ({x:.2f}, {y:.2f})"
            list_item = QListWidgetItem(item_text)
            self.ui.listWidget_Result.addItem(list_item)
