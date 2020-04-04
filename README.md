# Face Recognition ASSIGNMENT 1
Hana Nazmy 4952
Noha Nomier 4638	
Fatma Sherif 4701 

# Problem Statement
We intend to perform face recognition. Face recognition means that for a given image you can tell the subject id. Our database of subject is very simple. It has 40 subjects. Algorithms such as PCA and LDA are used to achieve the goal of the assignment.
Step 1: Reading the dataset
The dataset has 10 images per 40 subjects. Every image is a grayscale image of
size 92x112.
The dataset is saved using Google Drive, we use two arrays:
•	D of size 400 x 10304 to store 400 images
•	y of size 400 x 1 to store 400 labels for each image

The image is read using cv2.imread then reshaping it to a row vector and appending it to the D array.
Y array is also updated in each iteration with the corresponding image label which starts from 1 and ending with 40 (each label is repeated 10 times)
Step 2: Split the Dataset into Training and Test sets
The dataset is split in a way where 50% is for training and 50% for test. The corresponding label vector is split in the same way.
Training matrix corresponds to the even rows of the original D matrix and Test matrix corresponds to the odd rows of the original D matrix.
The resulting training and test matrices are of shape 200 x 10304 each.
The resulting label vectors are of size 200 x 1 each.
Further computations will be done using the training matrix to achieve best classification methods.

Step 3: Classification using PCA

PCA is a statistical approach used for reducing the number of variables in face recognition. In PCA, every image in the training set is represented as a linear combination of weighted eigenvectors called eigenfaces. These eigenvectors are obtained from covariance matrix of a training image set. The weights are found out after selecting a set of most relevant Eigenfaces. Recognition is performed by projecting a test image onto the subspace spanned by the eigenfaces and then classification is done by measuring minimum Euclidean distance.
•	Covariance matrix
First mean over training data is computed using numpy.mean(), the result is one image representing the mean of all images (1 x 10304)
Z matrix represents the data after being centered, this is done by subtracting the obtained mean from the training data :  Z=training-mean

Covariance matrix is obtained using np.cov() by passing the transpose of the Z matrix.
The covariance matrix Σ, which is a symmetric d×d so 10304 x 10304  matrix where each element represents the covariance between two features. 

•	EigenValues and EigenVectors
The eigendecomposition is performed on the covariance matrix Σ.
EigenValues and EigenVectors are obtained using numpy.linalg.eigh(covariance) which returns the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or a real symmetric matrix.
EigenValues are then sorted in the decreasing order using numpy.argsort() returning indices of the sorted array and then eigenvectors are sorted according to the same index.
This step is done because we need to choose the largest R eigenvalues which will give the best accuracy for classification.


•	Number of EigenVectors taken R
We are give 4 values of alpha  ={0.8,0.85,0.9,0.95}
For each alpha we will calculate the corresponding number of eigenvectors to use for projection.

Sum is the sum of all eigenvalues.
To get R for each alpha, a loop over the eigenvalues array is done, fractional_sum carries the sum of eigen_values until this iteration.
If fractional_sum/sum is bigger than or equal to alpha then we break out of the loop and corresponding R is the value of the n’th iteration.
r_array carries the four values of R for each alpha given


LDA: Linear Discriminant Analysis for face recognition
Linear Discriminant Analysis (LDA) is most commonly used as dimensionality reduction technique in the pre-processing step for pattern-classification and machine learning applications. The goal is to project a dataset onto a lower-dimensional space with good class-separability in order avoid overfitting and also reduce computational costs.
•	Calculating the mean per class
Lda_mean is an array (40x10304) that holds the mean for each class “person”.
•	Computing the between Scatter Matrix
First for computing the between scatter matrix we subtract the total dataset mean from the per class mean and save the output in the “diff” array temporarily.
Each row in the “diff” array is then reshaped to (1x10304) and the between scatter matrix is computed by this given rule: 
 
Where the “diff” array is multiplied with its transpose and the number of elements in each class.
The final between scatter matrix is of size ( 10304 x 10304 ). 
•	To Compute the Within Scatter Matrix
For the within scatter matrix, The data is centered by subtracting each image by its class’ mean, then the resulting data is transposed and dot product is performed according to this equation :
Si = Zi.T (dot) Zi
Where the within Scatter matrix is the summation of Si for each Class from i=1 to 40.
The Eigenvalues and Eigenvectors are then computed for the output matrix Inverse(Si) dot Sb.
We sort the eigenvalues in decreasing order and take the first 39 eigenvalues with the largest magnitudes (The most dominant eigenvalues) and their corresponding eigenvectors.

To reduce the training data, we project our training data and our test data onto the 39 eigenvectors.
  evecs_reduced =  np.real(sorted_lda_evecs[:,0:39] )
  reduced_lda_training= np.dot(training, evecs_reduced )
  reduced_lda_test=np.dot(test,  evecs_reduced )
  
•	To Find the K nearest Neighbours:
The k-nearest neighbors classifier is then used with the values 1,3,5,7 for the K parameter
for k in [1,3,5,7]:
    classifier_lda = KNeighborsClassifier(n_neighbors=k)
    classifier_lda.fit(reduced_lda_training,y_training.ravel(order='C'))
    predictions_lda = classifier_lda.predict(reduced_lda_test)
    acc_lda = accuracy_score(y_test,predictions_lda)





 
