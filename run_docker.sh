docker run -it -v $HOME:/data -v $(pwd):/workspace --privileged --rm --device=/dev/kfd --device=/dev/dri --group-add video commtf
