import matplotlib.pyplot as plt
import numpy as np

class Game2048Viewer:
    def __init__(self, name, board_size):
        self.name = name
        self.board_size = board_size

    def render(self, state, save=True, path="./2048.png"):
        board = np.array(state.board)
        fig, ax = plt.subplots(figsize=(self.board_size, self.board_size))

        # Define colors for each tile value
        tile_colors = {
            0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
            16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
            256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"
        }

        for (i, j), value in np.ndenumerate(board):
            color = tile_colors.get(int(value), "#3c3a32")
            rect = plt.Rectangle([j, i], 1, 1, facecolor=color, edgecolor="black")
            ax.add_patch(rect)
            if value != 0:
                ax.text(j + 0.5, i + 0.5, str(int(value)), va="center", ha="center", color="black", fontsize=16, fontweight='bold')

        ax.set_xlim(0, self.board_size)
        ax.set_ylim(0, self.board_size)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close(fig)

    def animate(self, states, interval=200, save_path=None):
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(self.board_size, self.board_size))

        tile_colors = {
            0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
            16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
            256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"
        }

        def draw_board(state):
            ax.clear()
            board = np.array(state.board)
            for (i, j), value in np.ndenumerate(board):
                color = tile_colors.get(int(value), "#3c3a32")
                rect = plt.Rectangle([j, i], 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)
                if value != 0:
                    ax.text(j + 0.5, i + 0.5, str(int(value)), va="center", ha="center", color="black", fontsize=16, fontweight='bold')
            ax.set_xlim(0, self.board_size)
            ax.set_ylim(0, self.board_size)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.gca().invert_yaxis()
            plt.tight_layout()

        ani = animation.FuncAnimation(
            fig, lambda i: draw_board(states[i]), frames=len(states), interval=interval
        )
        if save_path:
            ani.save(save_path, writer="imagemagick")
        else:
            plt.show()
        plt.close(fig)
        return ani

if __name__ == "__main__":
    viewer = Game2048Viewer("2048", 4)
    viewer.animate(states=states, interval=400)