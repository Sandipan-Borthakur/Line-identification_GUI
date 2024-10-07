from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget, QTableWidget,
                             QTableWidgetItem, QSplitter,
                             QCheckBox, QHeaderView, QPushButton)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector
from matplotlib.figure import Figure
from astropy.io import fits
from scipy import signal
import os.path
import sys
import numpy as np
sys.path.append('/export/borthaku/Codes/PyReduce')
from tqdm import tqdm
from scipy.optimize import least_squares
import warnings

def gaussval2(x, a, mu, sig, const):
    return a * np.exp(-((x - mu) ** 2) / (2 * sig)) + const

def gaussfit2(x, y):
    gauss = gaussval2

    x = np.ma.compressed(x)
    y = np.ma.compressed(y)

    if len(x) == 0 or len(y) == 0:
        raise ValueError("All values masked")

    if len(x) != len(y):
        raise ValueError("The masks of x and y are different")

    # Find the peak in the center of the image
    weights = np.ones(len(y), dtype=y.dtype)
    midpoint = len(y) // 2
    weights[:midpoint] = np.linspace(0, 1, midpoint, dtype=weights.dtype)
    weights[midpoint:] = np.linspace(1, 0, len(y) - midpoint, dtype=weights.dtype)

    i = np.argmax(y * weights)
    p0 = [y[i], x[i], 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = least_squares(
            lambda c: gauss(x, *c, np.ma.min(y)) - y,
            p0,
            loss="soft_l1",
            bounds=(
                [min(np.ma.mean(y), y[i]), np.ma.min(x), 0],
                [np.ma.max(y) * 1.5, np.ma.max(x), len(x) / 2],
            ),
        )
        popt = list(res.x) + [np.min(y)]
    return popt

class LineList:
    dtype = np.dtype(
        (
            np.record,
            [
                (("wlc", "WLC"), ">f8"),  # Wavelength (before fit)
                (("wll", "WLL"), ">f8"),  # Wavelength (after fit)
                (("posc", "POSC"), ">f8"),  # Pixel Position (before fit)
                (("posm", "POSM"), ">f8"),  # Pixel Position (after fit)
                (("xfirst", "XFIRST"), ">i2"),  # first pixel of the line
                (("xlast", "XLAST"), ">i2"),  # last pixel of the line
                (
                    ("approx", "APPROX"),
                    "O",
                ),  # Not used. Describes the shape used to approximate the line. "G" for Gaussian
                (("width", "WIDTH"), ">f8"),  # width of the line in pixels
                (("height", "HEIGHT"), ">f8"),  # relative strength of the line
                (("order", "ORDER"), ">i2"),  # echelle order the line is found in
                ("flag", "?"),  # flag that tells us if we should use the line or not
            ],
        )
    )

    def __init__(self, lines=None):
        if lines is None:
            lines = np.array([], dtype=self.dtype)
        self.data = lines
        self.dtype = self.data.dtype

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_list(cls, wave, order, pos, width, height, flag):
        lines = [
            (w, w, p, p, p - wi / 2, p + wi / 2, b"G", wi, h, o, f)
            for w, o, p, wi, h, f in zip(wave, order, pos, width, height, flag)
        ]
        lines = np.array(lines, dtype=cls.dtype)
        return cls(lines)

def _fit_single_line(obs, center, width):
    low = int(center - width * 5)
    low = max(low, 0)
    high = int(center + width * 5)
    high = min(high, len(obs))

    section = obs[low:high]
    x = np.arange(low, high, 1)
    x = np.ma.masked_array(x, mask=np.ma.getmaskarray(section))
    coef = gaussfit2(x, section)
    return coef

def fit_lines(obs, lines):

    for i, line in tqdm(
            enumerate(lines), total=len(lines), leave=False, desc="Lines"
    ):
        if line["posm"] < 0 or line["posm"] >= obs.shape[1]:
            # Line outside pixel range
            continue
        if line["order"] < 0 or line["order"] >= len(obs):
            # Line outside order range
            continue

        try:
            coef = _fit_single_line(
                obs[int(line["order"])],
                line["posm"],
                line["width"],
            )
            lines[i]["posm"] = coef[1]
        except:
            # Gaussian fit failed, dont use line
            lines[i]["flag"] = False

    return lines

def create_image_from_lines(lines, ncol):
    max_order = int(np.max(lines["order"]))
    img = np.zeros((max_order + 1, ncol))
    for line in lines:
        if line["order"] < 0:
            continue
        if line["xlast"] < 0 or line["xfirst"] > ncol:
            continue
        first = int(max(line["xfirst"], 0))
        last = int(min(line["xlast"], ncol))
        img[int(line["order"]), first:last] = line["height"] * signal.windows.gaussian(last - first, line["width"])
    return img
def find_offset(offset_info_path, lines, thar_master):
    if os.path.exists(offset_info_path):
        offset = np.loadtxt(offset_info_path)
        offset = [int(k) for k in offset]
    else:
        img = create_image_from_lines(lines, ncol=2088)
        correlation = signal.correlate2d(thar_master, img, mode="same")
        offset_order, offset_x = np.unravel_index(np.argmax(correlation), correlation.shape)
        offset_order = offset_order - img.shape[0] // 2 + 1
        offset_x = offset_x - img.shape[1] // 2 + 1
        offset = [int(offset_order), int(offset_x)]
        np.savetxt(offset_info_path, np.c_[int(offset_order), int(offset_x)], delimiter=' ', fmt=['%d', '%d'])
    return offset
def apply_alignment_offset(lines, offset, select=None):
    lines["xfirst"][select] += offset[1]
    lines["xlast"][select] += offset[1]
    lines["posm"][select] += offset[1]
    lines["order"][select] += offset[0]
    return lines
class PlotCanvas(FigureCanvas):
    def __init__(self, thar_master, lines, table, index=0, parent=None, width=5, height=4, dpi=100):
        self.thar_master = thar_master
        self.lines = lines
        self.orders = self.lines['order']
        self.positions = self.lines['posm']
        self.xfirsts = self.lines['xfirst']
        self.xlasts = self.lines['xlast']
        self.wavelengths = lines['wlc']
        self.pixels = np.arange(0, self.thar_master.shape[1])
        self.hidden_lines = set()  # Set to store hidden lines (track indices)
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.table = table  # Store reference to the QTableWidget
        super(PlotCanvas, self).__init__(self.fig)
        # Plot based on index
        self.plot(index)

    def on_button_click(self,index):
        print(self.temporder, self.temppeak, self.tempxfirst, self.tempxlast)
        wpeak = 0  # Placeholder for wavelength (you can replace with actual value)
        wipeak = self.tempxlast - self.tempxfirst
        hpeak = 1  # Placeholder for height (can be calculated or set manually)
        test = [[wpeak, self.temporder, self.temppeak, wipeak, hpeak, True]]
        test = np.array(test).T
        new_line = LineList.from_list(*test)
        # Append the new line to the existing lines
        self.lines = np.append(self.lines, new_line)

        # Update the QTableWidget with the new line
        self.add_line_to_table(new_line)
        # self.plot(index)

    def add_line_to_table(self, new_line):
        """Add the new line to the QTableWidget."""
        current_row_count = self.table.rowCount()
        self.table.insertRow(current_row_count)  # Add a new row for the new line

        checkbox = QCheckBox()
        checkbox.setChecked(True)  # Checked by default
        checkbox.stateChanged.connect(
            lambda state, line_idx=current_row_count: self.toggle_line(self.temporder, line_idx, state == 2))
        self.table.setCellWidget(current_row_count, 0, checkbox)  # Add the checkbox to the 0th column

        # Populate the row with the new line data (order, wlc, posm, xfirst, xlast)
        keys = ['order', 'wlc', 'posm', 'xfirst', 'xlast']
        for col, key in enumerate(keys):
            item = QTableWidgetItem(f'{new_line[key][0]:.4f}' if col > 0 else f'{new_line[key][0]:.0f}')
            item.setFlags(item.flags() | Qt.ItemIsEditable)  # Make the cell editable
            self.table.setItem(current_row_count, col + 1, item)

    def toggle_line(self, index, line_idx, checked):
        """Toggle the visibility of a specific line based on checkbox state."""
        if checked:
            self.hidden_lines.discard(line_idx)  # Remove from hidden lines if checked
        else:
            self.hidden_lines.add(line_idx)  # Add to hidden lines if unchecked
        self.plot(index)  # Re-plot to reflect changes


    def onselect(self, xmin, xmax, line_idx, thar_master_single_order):
        self.temporder, self.tempxfirst, self.tempxlast = line_idx, int(xmin), int(xmax)
        flux_order = thar_master_single_order[self.tempxfirst:self.tempxlast+1]
        self.temppeak = np.argmax(flux_order) + self.tempxfirst

    def plot(self, index):
        self.button = QPushButton('Add line', self)
        self.button.clicked.connect(self.on_button_click)
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        self.axes.clear()
        self.axes.plot(np.arange(0, self.thar_master.shape[1]), self.thar_master[index])
        if index in self.orders:
            ind = np.where(self.orders == index)[0]
            croppositions = self.positions[ind]
            cropxfirsts = self.xfirsts[ind]
            cropxlasts = self.xlasts[ind]
            cropwavelengths = self.wavelengths[ind]
            croporders = self.orders[ind]
            for j in range(len(ind)):
                if j in self.hidden_lines:  # Skip if the line is hidden
                    continue
                indlinepos = np.argmax(self.thar_master[index][cropxfirsts[j]:cropxlasts[j] + 1])
                linepos = self.thar_master[index][cropxfirsts[j]:cropxlasts[j] + 1][indlinepos]
                maxpixel = self.pixels[cropxfirsts[j] + indlinepos]
                self.axes.plot([maxpixel, maxpixel], [1.1 * linepos, 1.5 * linepos], 'r', lw=2)
                self.axes.text(maxpixel, 1.6 * linepos, '{:.4f}'.format(cropwavelengths[j]), rotation='vertical')
        self.span = SpanSelector(
            self.axes,
            lambda xmin, xmax: self.onselect(xmin, xmax, line_idx=index, thar_master_single_order=self.thar_master[index]),
            "horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor="tab:green"),
            interactive=True,
            drag_from_anywhere=True
        )
        self.axes.set_title(f'ThAr Master Order {index + 1}')
        self.axes.set_xlabel('Pixels')
        self.axes.set_ylabel('Intensity')
        self.axes.set_ylim((-400, max(self.thar_master[index]) + 400))
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wavelength Calibration Line Identification")
        self.selected_lines = None  # Store the selected lines here

        tab_widget = QTabWidget()
        self.setCentralWidget(tab_widget)

        thar_master_path = '/export/borthaku/Codes/toes-pyreduce-pipeline/TOES-reduced/toes.thar_master.fits'
        hdu = fits.open(thar_master_path)
        self.thar_master = hdu[0].data

        linelist_path = '/export/borthaku/Codes/toes-pyreduce-pipeline/TOES-reduced/toes.linelist.npz'
        data = np.load(linelist_path, allow_pickle=True)
        self.lines = data['cs_lines']

        offset_info_path = 'offset_info.txt'
        offset = find_offset(offset_info_path, self.lines, self.thar_master)
        self.lines = apply_alignment_offset(self.lines, offset)

        for i in range(self.thar_master.shape[0]):
            tab = QWidget()
            layout = QVBoxLayout(tab)

            linelist_table = QTableWidget()
            canvas = PlotCanvas(self.thar_master, self.lines, linelist_table, index=i, parent=self, width=5, height=4)

            toolbar = NavigationToolbar(canvas, self)

            self.populate_linelist_table(linelist_table, self.lines, i, canvas)

            splitter = QSplitter()
            splitter.addWidget(canvas)
            splitter.addWidget(linelist_table)
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 1)

            layout.addWidget(splitter)
            layout.addWidget(toolbar)
            tab_widget.addTab(tab, f"Order {i + 1}")

            self.button = QPushButton('Save', self)
            self.button.clicked.connect(self.onclick)
            layout.addWidget(self.button)

    def populate_linelist_table(self, table, lines, index, canvas):
        """Populate the QTableWidget with linelist data and add checkboxes to toggle line visibility."""
        order_lines = lines[lines['order'] == index]
        table.setRowCount(len(order_lines))
        table.setColumnCount(6)  # 1 for checkbox, 5 for data columns
        table.setHorizontalHeaderLabels(['Select', 'Order', 'Wavelength', 'Position', 'xFirst', 'xLast'])

        for i, line in enumerate(order_lines):
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # Checked by default
            checkbox.stateChanged.connect(
                lambda state, line_idx=i: self.toggle_line(canvas, index, line_idx, state == 2))
            table.setCellWidget(i, 0, checkbox)

            for col, key in enumerate(['order', 'wlc', 'posm', 'xfirst', 'xlast']):
                item = QTableWidgetItem(f'{line[key]:.4f}' if col > 0 else f'{line[key]:.0f}')
                item.setFlags(item.flags() | Qt.ItemIsEditable)  # Make the cell editable
                table.setItem(i, col + 1, item)

    def toggle_line(self, canvas, index, line_idx, checked):
        """Toggle line visibility when checkbox is checked/unchecked."""
        canvas.toggle_line(index, line_idx, checked)

    def onclick(self):
        """Save button functionality."""
        # Collect the visible lines
        visible_lines = self.get_visible_lines()

        # Save the visible lines to an npz file
        save_path = os.path.join(os.getcwd(), 'linelist_test.npz')
        np.savez(save_path, cs_lines=visible_lines)

        print(f'Selected lines saved to {save_path}')

    def get_visible_lines(self):
        visible_lines = []

        for i in range(self.centralWidget().count()):  # Loop over all the tabs
            # Access the QSplitter in the tab (it contains both the PlotCanvas and the table)
            splitter = self.centralWidget().widget(i).layout().itemAt(0).widget()  # The splitter is the first widget
            canvas = splitter.widget(0)  # The PlotCanvas is the first widget in the splitter

            for j, line in enumerate(canvas.lines[canvas.lines['order']==i]):
                if j not in canvas.hidden_lines:  # Check if the line is visible
                    visible_lines.append(line)
        return np.array(visible_lines)

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.showMaximized()
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()