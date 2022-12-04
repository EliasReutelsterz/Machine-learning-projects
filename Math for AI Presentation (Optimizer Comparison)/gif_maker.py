import imageio

class GifMaker():
    def makegif(iterations: int, start_time: str) -> None:
        images = []
        filenames = [f'figures/{start_time}/{start_time}' + '-' + str(i) + '.png' for i in range(iterations)]
        imageio.plugins.freeimage.download()
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'figures/{start_time}/movie({start_time}).gif', images, format='GIF-FI', duration=0.001)