#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"

/* IDX format: big-endian 32-bit header fields, then raw unsigned bytes.
 *   images: magic 0x00000803, count, rows, cols, pixels...
 *   labels: magic 0x00000801, count, labels...
 */
static int read_be32(FILE *fp, unsigned int *out) {
    unsigned char buf[4];
    if (fread(buf, 1, 4, fp) != 4) return 0;
    *out = ((unsigned int)buf[0] << 24) | ((unsigned int)buf[1] << 16) |
           ((unsigned int)buf[2] << 8) | (unsigned int)buf[3];
    return 1;
}

MnistSet *mnist_load(const char *image_path, const char *label_path, int max_count) {
    FILE *fi = fopen(image_path, "rb");
    FILE *fl = fopen(label_path, "rb");
    if (!fi || !fl) {
        fprintf(stderr, "Error: could not open MNIST files '%s' / '%s'.\n"
                        "       Run `make data` (or copy the files into data/).\n",
                image_path, label_path);
        if (fi) fclose(fi);
        if (fl) fclose(fl);
        return NULL;
    }

    unsigned int magic_i, n_i, rows, cols, magic_l, n_l;
    if (!read_be32(fi, &magic_i) || !read_be32(fi, &n_i) ||
        !read_be32(fi, &rows) || !read_be32(fi, &cols) ||
        !read_be32(fl, &magic_l) || !read_be32(fl, &n_l) ||
        magic_i != 0x803 || magic_l != 0x801 ||
        rows * cols != MNIST_PIXELS || n_i != n_l) {
        fprintf(stderr, "Error: '%s' / '%s' are not matching MNIST IDX files.\n",
                image_path, label_path);
        fclose(fi);
        fclose(fl);
        return NULL;
    }

    int count = (int)n_i;
    if (max_count > 0 && max_count < count) count = max_count;

    MnistSet *set = malloc(sizeof(MnistSet));
    unsigned char *raw = malloc((size_t)count * MNIST_PIXELS);
    unsigned char *raw_labels = malloc((size_t)count);
    set->count = count;
    set->images = malloc((size_t)count * MNIST_PIXELS * sizeof(float));
    set->labels = malloc((size_t)count * sizeof(int));
    if (!set->images || !set->labels || !raw || !raw_labels) {
        fprintf(stderr, "Error: out of memory loading MNIST\n");
        exit(1);
    }

    if (fread(raw, 1, (size_t)count * MNIST_PIXELS, fi) != (size_t)count * MNIST_PIXELS ||
        fread(raw_labels, 1, (size_t)count, fl) != (size_t)count) {
        fprintf(stderr, "Error: MNIST files are truncated.\n");
        exit(1);
    }

    for (long k = 0; k < (long)count * MNIST_PIXELS; k++) {
        set->images[k] = raw[k] / 255.0f;
    }
    for (int k = 0; k < count; k++) {
        set->labels[k] = raw_labels[k];
    }

    free(raw);
    free(raw_labels);
    fclose(fi);
    fclose(fl);
    return set;
}

void mnist_free(MnistSet *set) {
    if (!set) return;
    free(set->images);
    free(set->labels);
    free(set);
}
