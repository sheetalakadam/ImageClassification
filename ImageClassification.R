# Read binary file and convert to integer vectors
# [Necessary because reading directly as integer() 
# reads first bit as signed otherwise]
#
# File format is 10000 records following the pattern:
# [label x 1][red x 1024][green x 1024][blue x 1024]
# NOT broken into rows, so need to be careful with "size" and "n"
#

labels <- read.table("https://personal.utdallas.edu/~sak170006/batches.meta.txt")
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

