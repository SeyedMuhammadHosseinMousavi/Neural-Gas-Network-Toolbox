# Neural-Gas-Network-Toolbox
Neural Gas Network (NGN) Toolbox - NGN Feature Selection - Supervised NGN (NGN Classifier) - NGN Data Augmentor 

The Neural Gas Network (NGN) is an artificial neural network algorithm introduced by Thomas Martinetz and Klaus Schulten in 1991, designed to perform vector quantization and clustering by adapting its structure to the data topology. Unlike the Self-Organizing Map (SOM), which imposes a predefined grid structure, the NGN flexibly arranges its neurons in the data space based on a competitive learning process. Each neuron (or unit) in the NGN moves towards input data points, with the amount of movement influenced by both the distance to the data point and the rank order of the neuron. This process minimizes a cost function related to the distances between the neurons and the data points, effectively capturing the underlying distribution and structure of the data. The NGN is particularly useful for tasks requiring unsupervised learning, such as clustering, vector quantization, and dimensionality reduction, and it can be adapted for supervised learning tasks or synthetic data generation by incorporating class labels and specific update rules.

The Neural Gas Network (NGN) can be effectively utilized for feature selection by leveraging its ability to adaptively learn the underlying structure of high-dimensional data. During the training process, the NGN adjusts its neurons to minimize the distance to input data points, and the extent of these adjustments can be tracked to measure the influence of each feature. By analyzing the magnitude of changes in neuron positions relative to each feature, we can determine which features contribute most significantly to the data representation. This approach provides a way to rank and select the most informative features, thereby reducing dimensionality and improving the performance of subsequent machine learning models. Feature selection with NGN is particularly beneficial for datasets with many irrelevant or redundant features, as it helps in identifying and retaining only the most impactful features, leading to more efficient and effective learning.

The Supervised Neural Gas (SNG) classifier extends the original Neural Gas Network (NGN) by incorporating supervised learning principles to enable classification tasks. In the SNG classifier, neurons are initialized and updated not only based on their proximity to input data points but also considering the class labels of the data. During training, each neuron adapts its position in the feature space and its associated class label based on the input data points and their respective labels, resulting in a topology-preserving mapping that reflects both the data distribution and class separations. This supervised adaptation allows the SNG classifier to effectively distinguish between different classes in the data. After training, new data points are classified by identifying the nearest neuron and assigning its label to the input, leveraging the learned class-specific topologies. The SNG classifier is particularly useful for applications where maintaining the topological structure of the data while accurately classifying it is crucial, such as in pattern recognition, image classification, and anomaly detection.

The Neural Gas Network (NGN) can also be employed for synthetic data generation (SDG) by leveraging its ability to learn the distribution and topology of the input data through its adaptive neuron placement. Once trained, the NGN represents the data distribution with a set of neurons, each positioned to capture the essential characteristics of the input space. Synthetic data can be generated by sampling around these neurons, introducing controlled variations or noise to create new data points that mimic the properties of the original dataset. This approach ensures that the generated synthetic data maintains the same statistical properties and structural relationships as the real data. Using NGN for SDG is particularly valuable in scenarios where data augmentation is needed to enhance model training, address class imbalances, or simulate new scenarios for testing and validation. The flexibility and adaptability of NGN make it a powerful tool for generating high-quality synthetic data that can improve the robustness and generalization of machine learning models.

![ngn](https://github.com/SeyedMuhammadHosseinMousavi/Neural-Gas-Network-Toolbox/assets/11339420/e2d43c4e-0901-403f-ad21-2251a1bd3b54)

Using the Neural Gas Network (NGN) for image segmentation represents a novel approach to tackling the task of dividing an image into meaningful segments. The NGN is an unsupervised learning algorithm that adapts to the underlying distribution of data points, and in the context of image segmentation, these data points are the pixels of an image. Each pixel can be considered a multi-dimensional point where dimensions correspond to color channels and potentially spatial coordinates. By training the NGN on pixel data, the network units move and adapt to form clusters that correspond to different color regions or textures within the image. This allows the NGN to effectively identify distinct areas based on similarities in color and texture, segmenting the image into coherent regions. The strength of NGN lies in its ability to form a topological map of the input space, capturing intrinsic patterns and relationships that are useful for segmentation tasks. This method is particularly advantageous for complex images where traditional segmentation techniques might struggle to accurately differentiate between regions due to subtle differences in texture or color gradients.

![NGN Segmentation](https://github.com/SeyedMuhammadHosseinMousavi/Neural-Gas-Network-Toolbox/assets/11339420/e230de15-d08d-4de2-b4a5-ff92c4bf17d6)

Using Neural Gas Networks (NGN) for topology fitting is an innovative application of this unsupervised learning algorithm, particularly valuable in understanding and modeling complex data structures. Unlike many traditional clustering algorithms that only focus on grouping similar data points, NGNs excel in adapting to the overall distribution of data, preserving the topological properties of the input space. This capability makes them highly suitable for topology fitting, where the goal is to map high-dimensional data onto a lower-dimensional space while maintaining the intrinsic relationships between data points. As NGN iteratively adjusts its units towards data points based on both proximity and rank order within the neighborhood, it effectively learns the shape and structure of the data. This process not only aids in visualizing high-dimensional data in a more comprehensible form but also enhances the analysis of data dynamics, patterns, and clusters, providing deeper insights into the underlying structures that govern the data. In practical applications, topology fitting via NGN can be pivotal in fields such as computational neuroscience, complex systems analysis, and advanced data visualization, where understanding the interconnections and layout of data is crucial.

![NGN Topology](https://github.com/SeyedMuhammadHosseinMousavi/Neural-Gas-Network-Toolbox/assets/11339420/b2943392-cafc-4ca1-969b-6ee6ed07b0f3)
