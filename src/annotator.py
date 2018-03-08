import matplotlib.pyplot as plt
from os import path

class Annotator:
    """
    Object used to manually label interest points in pairs of objects
    Example usage:
    >>> ann = Annotator(<im1>, <im2>, check_for_annotations = True)
        - creates a new object. It is by default in annotation mode
        - check_for_annotations will force the check of previously 
        labelled interest points in the same directory as annotator.py
    
    >>> ann.annotate()
        - shows the plot window containing both images.
        labelling starts by clicking on an interest point in the leftmost image
        then click on the associated point in the right image.
        Do this for however many interest points are needed
        - close plot window to finish labelling

    >>> ann.save_annotations()
        - saves the new labelling into files linked to the images' filenames

    """
    def __init__(self, path1, path2, annotation_mode = True, check_for_annotations = True):
        self.image_paths = [path1, path2]
        self.images = [
            plt.imread(p) for p in self.image_paths
            ]

        if annotation_mode:
            print('Loading Annotator in annotation mode...')
            self.current_image = 0
            
            if check_for_annotations:
                self.coords = self.__look_for_annotations()
            else:
                self.coords = [[], []]

            self.display_images()
            self.cid = self.fig.canvas.mpl_connect('button_press_event', self.__onClick)

        else:
            print('Loading Annotator in data load mode...')
            self.coords = self.__look_for_annotations()

    def display_images(self, show = False):
        self.fig, self.axes = plt.subplots(1, 2)
        self.axes[0].imshow(self.images[0], cmap='gray')
        self.axes[1].imshow(self.images[1], cmap='gray')

        if self.coords[0]:
            x, y = zip(*self.coords[0])
            self.axes[0].scatter(x, y, marker='x', c='y')
        if self.coords[1]:
            x, y = zip(*self.coords[1])
            self.axes[1].scatter(x, y, marker='x', c='y')

        if show:
            plt.show()

    def __onClick(self, event):

        ix, iy = event.xdata, event.ydata
        if ix is not None and iy is not None:
            ix, iy = int(ix), int(iy)
            print('Image: {}, x = {}, y = {}'.format(self.current_image, ix, iy))
            self.coords[self.current_image].append([ix, iy])

            self.axes[self.current_image].scatter([ix], [iy], marker='x', c='r')
            self.fig.canvas.draw()

            self.current_image = int(not self.current_image)

    def annotate(self):
        plt.show()

    def save_annotations(self):
        annotation_name = self.get_annotation_name()

        print('> Saving annotations to {}...'.format(annotation_name))
        with open(annotation_name, 'w') as f:
            f.write('{}\n'.format(len(self.coords[0])))
            for i in range(len(self.image_paths)):
                f.write(self.image_paths[i] + '\n')
                string_coords = '\n'.join([' '.join(list(map(str, c))) for c in self.coords[i]])
                f.write(string_coords + '\n')

    def __look_for_annotations(self):
        annot = self.get_annotation_name()

        print('Image Files:\n> {}\n> {}\nSearching for annotation file: {}'.format(
            self.image_paths[0], self.image_paths[1], annot))

        coords = [[], []]

        if path.exists(annot):
            print('> Annotation file found.')

            with open(annot, 'r') as f:
                try:
                    data = f.read().split('\n')
                    if data[-1] == '':
                        data = data[:-1]

                    num = int(data[0])
                    coords[0] = [list(map(int, c.split(' '))) for c in data[2 : num+2]]
                    coords[1] = [list(map(int, c.split(' '))) for c in data[num + 3 :]]

                except ValueError:
                    pass # silently fail as file is probably empty
                    
        else:
            print('> WARNING: Annotation file not found.')

        return coords

    def get_annotation_name(self):
        names = ['.'.join(path.basename(p).split('.')[:-1]) for p in self.image_paths]
        return '{}_{}_annotation.txt'.format(names[0], names[1])


if __name__ == '__main__':
    
    # ann = Annotator('edge.PNG', 'edge.1.PNG', check_for_annotations = True) # create annotator object
    # ann.annotate() # begin annotating images
    # ann.save_annotations() # write annotations to disk

    # ann = Annotator('edge.PNG', 'edge.1.PNG', annotation_mode = False) # create annotator object in simple display mode 
    # ann.display_images(show=True) # display images + annotations

    import sys
    ann = Annotator(sys.argv[1], sys.argv[2], check_for_annotations = True) # create annotator object
    ann.annotate() # begin annotating images
    ann.save_annotations() # write annotations to disk