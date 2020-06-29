#Author:
#SheetalKadam (sak170006)
#Darrel Donald (dld180001)

#Desc: Image classification of CIFAR-10 image dataset

#install and load libraries

load.libraries <-
  c('h2o',
    'tidyverse',
    'gridExtra',
    'grid')

libraries_to_install <-
  load.libraries[!load.libraries %in% installed.packages()]
for (lib in libraries_to_install)
  install.packages(lib, dependences = TRUE)
sapply(load.libraries, require, character = TRUE)

# data exploration

labels <-
  read.table("https://personal.utdallas.edu/~sak170006/batches.meta.txt")
images.rgb <- list()
images.lab <- list()
num.images = 10000 # Set to 10000 to retrieve all images per file to memory

# Cycle through all 1 of the binary files

for (f in 1:1) {
  to.read <-
    file(
      paste(
        "https://personal.utdallas.edu/~sak170006//data_batch_",
        f,
        ".bin",
        sep = ""
      ),
      "rb"
    )
  for (i in 1:num.images) {
    l <- readBin(to.read,
                 integer(),
                 size = 1,
                 n = 1,
                 endian = "big")
    r <-
      as.integer(readBin(
        to.read,
        raw(),
        size = 1,
        n = 1024,
        endian = "big"
      ))
    g <-
      as.integer(readBin(
        to.read,
        raw(),
        size = 1,
        n = 1024,
        endian = "big"
      ))
    b <-
      as.integer(readBin(
        to.read,
        raw(),
        size = 1,
        n = 1024,
        endian = "big"
      ))
    index <- num.images * (f - 1) + i
    images.rgb[[index]] = data.frame(r, g, b)
    images.lab[[index]] = l + 1
  }
  close(to.read)
  remove(l, r, g, b, f, i, index, to.read)
}

#funcion to display image

drawImage <- function(index) {
  img <- images.rgb[[index]]
  img.r.mat <- matrix(img$r, ncol = 32, byrow = TRUE)
  img.g.mat <- matrix(img$g, ncol = 32, byrow = TRUE)
  img.b.mat <- matrix(img$b, ncol = 32, byrow = TRUE)
  img.col.mat <-
    rgb(img.r.mat, img.g.mat, img.b.mat, maxColorValue = 255)
  dim(img.col.mat) <- dim(img.r.mat)
  
  grid.raster(img.col.mat, interpolate = FALSE)
  # clean up
  remove(img, img.r.mat, img.g.mat, img.b.mat, img.col.mat)
  
  labels[[1]][images.lab[[index]]]
}

drawImage(10)

#setwd("C:\\SKDrive\\UT Dallas\\Courses\\Sem4- Summer20\\Practical Aspects of DS\\Projects\\pROJECT2")
sample_images = c(1, 45, 96, 200, 1000)


#start H2o
h2o.init(
  ip = "localhost",
  port = 54321,
  nthreads = -1,
  min_mem_size = "20g"
)

#import data
cifar_data <-
  h2o.importFile(
    path = "https://personal.utdallas.edu/~sak170006//cifar_10.csv",
    destination_frame = "cifar_data",
    col.types = c(rep("factor", 1025))
  )

df <- data.frame(id = 1:60000)
temp <- as.h2o(df)
h2o.ls()
# create id
cifar <- h2o.cbind(temp, cifar_data)

cifar.split <- h2o.splitFrame(data = cifar, ratios = 0.75)
cifar_train <- cifar.split[[1]]

# Create a testing set from the 2nd dataset in the split
cifar_test <- cifar.split[[2]]

h2o.ls()

#test models with different paramters
#iteration 1
cifar_nn_1 <- h2o.deeplearning(
  x = 2:1025,
  y = "y",
  training_frame = cifar_train,
  validation_frame = cifar_test,
  model_id = "cifar_nn_1",
  l2 = 0.4,
  ignore_const_cols = FALSE,
  hidden = 10
)

h2o.confusionMatrix(cifar_nn_1, cifar_train)
h2o.confusionMatrix(cifar_nn_1, cifar_test)

plot(cifar_nn_1, timestep = "epochs", metric = "logloss")
plot(cifar_nn_1, timestep = "epochs",  metric = "classification_error")

predicted <- h2o.predict(cifar_nn_1, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}


#iteration 2
cifar_nn_2 <- h2o.deeplearning(
  x = 1:1024,
  y = "y",
  training_frame = cifar_train,
  validation_frame = cifar_test,
  distribution = "multinomial",
  model_id = "cifar_nn_2",
  activation = "RectifierWithDropout",
  hidden = c(10, 10, 10),
  epochs = 10,
  loss = "CrossEntropy",
  fast_mode = TRUE,
  diagnostics = TRUE,
  ignore_const_cols = TRUE,
  force_load_balance = TRUE,
  seed = 300
)


h2o.confusionMatrix(cifar_nn_2, cifar_train) #1-.73
h2o.confusionMatrix(cifar_nn_2, cifar_test) #.3

plot(cifar_nn_2, timestep = "epochs", metric = "logloss")
plot(cifar_nn_2, timestep = "epochs",  metric = "classification_error")

predicted <- h2o.predict(cifar_nn_2, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}

#iteration 3
cifar_nn_3 <- h2o.deeplearning(
  x = 1:1024,
  y = "y",
  training_frame = cifar_train,
  validation_frame = cifar_test,
  distribution = "multinomial",
  model_id = "cifar_nn_3",
  activation = "tanh",
  hidden = c(10, 10, 10),
  epochs = 10,
  loss = "CrossEntropy",
  fast_mode = TRUE,
  diagnostics = TRUE,
  ignore_const_cols = TRUE,
  force_load_balance = TRUE,
  seed = 300
)


h2o.confusionMatrix(cifar_nn_3, cifar_train) #1-.0018
h2o.confusionMatrix(cifar_nn_3, cifar_test) #.1-0.78

plot(cifar_nn_3, timestep = "epochs", metric = "logloss")
plot(cifar_nn_3, timestep = "epochs",  metric = "classification_error")

predicted <- h2o.predict(cifar_nn_3, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}

#iteration 4

cifar_nn_4 <- h2o.deeplearning(
  x = 1:1024,
  y = "y",
  training_frame = cifar_train,
  validation_frame = cifar_test,
  distribution = "multinomial",
  model_id = "cifar_nn_4",
  activation = "tanh",
  hidden = c(5, 5, 5, 5, 5),
  epochs = 10,
  loss = "CrossEntropy",
  fast_mode = TRUE,
  diagnostics = TRUE,
  ignore_const_cols = TRUE,
  force_load_balance = TRUE,
  seed = 300
)


h2o.confusionMatrix(cifar_nn_4, cifar_train) #1-.05
h2o.confusionMatrix(cifar_nn_4, cifar_test)  #1-.8

plot(cifar_nn_4, timestep = "epochs", metric = "logloss")
plot(cifar_nn_4, timestep = "epochs",  metric = "classification_error")

predicted <- h2o.predict(cifar_nn_4, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}

#iteration 5

cifar_nn_5 <- h2o.deeplearning(
  x = 1:1024,
  y = "y",
  training_frame = cifar_train,
  validation_frame = cifar_test,
  distribution = "multinomial",
  model_id = "cifar_nn_5",
  activation = "RectifierWithDropout",
  hidden = c(10, 10, 10),
  epochs = 20,
  loss = "CrossEntropy",
  fast_mode = TRUE,
  diagnostics = TRUE,
  ignore_const_cols = TRUE,
  force_load_balance = TRUE,
  seed = 300
)



h2o.confusionMatrix(cifar_nn_5, cifar_train) #1-.77
h2o.confusionMatrix(cifar_nn_5, cifar_test) #1-0.85

plot(cifar_nn_5, timestep = "epochs", metric = "logloss")
plot(cifar_nn_5, timestep = "epochs",  metric = "classification_error")


predicted <- h2o.predict(cifar_nn_5, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}

#iteration 6
cifar_nn_6 <- h2o.deeplearning(
  x = 1:1024,
  y = "y",
  training_frame = cifar_train,
  validation_frame = cifar_test,
  distribution = "multinomial",
  model_id = "cifar_nn_6",
  activation = "RectifierWithDropout",
  hidden = c(10, 10, 10, 10),
  epochs = 60,
  loss = "CrossEntropy",
  fast_mode = TRUE,
  diagnostics = TRUE,
  ignore_const_cols = TRUE,
  force_load_balance = TRUE,
  seed = 300
)



h2o.confusionMatrix(cifar_nn_6, cifar_train) #1-.85
h2o.confusionMatrix(cifar_nn_6, cifar_test) #1-0.84

plot(cifar_nn_6, timestep = "epochs", metric = "logloss")
plot(cifar_nn_6, timestep = "epochs",  metric = "classification_error")


predicted <- h2o.predict(cifar_nn_6, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}

#iteration 7
cifar_nn_7 <- h2o.deeplearning(
  x = 1:1024,
  y = "y",
  training_frame = cifar_train,
  validation_frame = cifar_test,
  distribution = "multinomial",
  model_id = "cifar_nn_7",
  activation = "tanh",
  hidden = c(10, 10, 10),
  epochs = 2,
  loss = "CrossEntropy",
  fast_mode = TRUE,
  diagnostics = TRUE,
  ignore_const_cols = TRUE,
  force_load_balance = TRUE,
  seed = 300
)


h2o.confusionMatrix(cifar_nn_7, cifar_train) #1-.18
h2o.confusionMatrix(cifar_nn_7, cifar_test) #1-0.8


plot(cifar_nn_7, timestep = "epochs", metric = "logloss")
plot(cifar_nn_7, timestep = "epochs",  metric = "classification_error")

predicted <- h2o.predict(cifar_nn_7, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}

#iteration 8
cifar_nn_8 <- h2o.deeplearning(
  x = 1:1024,
  y = "y",
  training_frame = cifar_train,
  validation_frame = cifar_test,
  distribution = "multinomial",
  model_id = "cifar_nn_8",
  activation = "tanh",
  hidden = c(10, 10),
  epochs = 1,
  loss = "CrossEntropy",
  fast_mode = TRUE,
  diagnostics = TRUE,
  ignore_const_cols = TRUE,
  force_load_balance = TRUE,
  seed = 300
)


h2o.confusionMatrix(cifar_nn_8, cifar_train) #1-.32
h2o.confusionMatrix(cifar_nn_8, cifar_test) #1-0.75

plot(cifar_nn_8, timestep = "epochs", metric = "logloss")
plot(cifar_nn_8, timestep = "epochs",  metric = "classification_error")


predicted <- h2o.predict(cifar_nn_8, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}


#iteration 9

cifar_nn_9 <- h2o.deeplearning(
  x = 1:1024,
  y = "y",
  training_frame = cifar_train,
  distribution = "multinomial",
  validation_frame = cifar_test,
  model_id = "cifar_nn_9",
  activation = "Maxout",
  hidden = c(10, 10),
  epochs = 1,
  fast_mode = TRUE,
  diagnostics = TRUE,
  ignore_const_cols = TRUE,
  force_load_balance = TRUE,
  seed = 300
)


h2o.confusionMatrix(cifar_nn_9, cifar_train) #1-.30
h2o.confusionMatrix(cifar_nn_9, cifar_test) #1-0.74

plot(cifar_nn_9, timestep = "epochs", metric = "logloss")
plot(cifar_nn_9, timestep = "epochs",  metric = "classification_error")

predicted <- h2o.predict(cifar_nn_9, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}

#iteration 10

cifar_nn_10 <- h2o.deeplearning(
  x = 1:1024,
  y = "y",
  training_frame = cifar_train,
  validation_frame = cifar_test,
  distribution = "multinomial",
  model_id = "cifar_nn_10",
  activation = "Maxout",
  hidden = c(10, 10, 10, 10, 10),
  epochs = 2,
  seed = 300
)


h2o.confusionMatrix(cifar_nn_10, cifar_train) #1-.17
h2o.confusionMatrix(cifar_nn_10, cifar_test) #1-0.74

plot(cifar_nn_10, timestep = "epochs", metric = "logloss")
plot(cifar_nn_10, timestep = "epochs",  metric = "classification_error")

predicted <- h2o.predict(cifar_nn_10, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}

#iteration 11
cifar_nn_11 <- h2o.deeplearning(
  x = 1:1024,
  y = "y",
  training_frame = cifar_train,
  validation_frame = cifar_test,
  distribution = "multinomial",
  model_id = "cifar_nn_11",
  hidden = c(10, 10, 10, 10, 10),
  epochs = 2,
  seed = 200
)


h2o.confusionMatrix(cifar_nn_11, cifar_train) #1-.38
h2o.confusionMatrix(cifar_nn_11, cifar_test) #1-0.78

plot(cifar_nn_11, timestep = "epochs", metric = "logloss")
plot(cifar_nn_11, timestep = "epochs",  metric = "classification_error")

predicted <- h2o.predict(cifar_nn_11, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}

#iteration 12
cifar_nn_12 <- h2o.deeplearning(
  x = 2:1025,
  y = "y",
  training_frame = cifar_train,
  validation_frame = cifar_test,
  distribution = "multinomial",
  model_id = "cifar_nn_12",
  activation = "RectifierWithDropout",
  hidden = c(40, 50, 10, 10, 10),
  epochs = 15,
  loss = "CrossEntropy",
  seed = 300
)

h2o.confusionMatrix(cifar_nn_12, cifar_train) #1-.88
h2o.confusionMatrix(cifar_nn_12, cifar_test)

plot(cifar_nn_12, timestep = "epochs", metric = "logloss")
plot(cifar_nn_12, timestep = "epochs",  metric = "classification_error")


predicted <- h2o.predict(cifar_nn_12, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}

#iteration 13
cifar_nn_13 <- h2o.deeplearning(
  x = 2:1025,
  y = "y",
  training_frame = cifar_train,
  validation_frame = cifar_test,
  distribution = "multinomial",
  model_id = "cifar_nn_13",
  activation = "RectifierWithDropout",
  hidden = c(40, 50, 10, 10, 10),
  epochs = 5,
  loss = "CrossEntropy",
  seed = 300
)

h2o.confusionMatrix(cifar_nn_13, cifar_train) #1-.88
h2o.confusionMatrix(cifar_nn_13, cifar_test) #1-.88

plot(cifar_nn_13, timestep = "epochs", metric = "logloss")
plot(cifar_nn_13, timestep = "epochs",  metric = "classification_error")

predicted <- h2o.predict(cifar_nn_13, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}


#iteration 14

cifar_nn_14 <- h2o.deeplearning(
  x = 1:1024,
  y = "y",
  training_frame = cifar_train,
  validation_frame = cifar_test,
  model_id = "cifar_nn_14",
  l2 = 0.4,
  ignore_const_cols = FALSE,
  hidden = 10,
  export_weights_and_biases = TRUE
)

h2o.confusionMatrix(cifar_nn_14, cifar_train) #1-.88
h2o.confusionMatrix(cifar_nn_14, cifar_test) #1-.88

plot(cifar_nn_14, timestep = "epochs", metric = "logloss")
plot(cifar_nn_14, timestep = "epochs",  metric = "classification_error")


predicted <- h2o.predict(cifar_nn_14, cifar_test)
pred_vs_actual <-
  as.data.frame(h2o.cbind(cifar_test$id, cifar_test$y, predicted$predict))
for (f in sample_images) {
  print(pred_vs_actual[f,])
}


h2o.shutdown()
