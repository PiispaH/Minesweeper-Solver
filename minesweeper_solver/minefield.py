from enum import Enum
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class GameState(Enum):
    """Enumeration for the possible states of the cells"""

    PLAYING = "facesmile"
    LOST = "facedead"
    WIN = "facewin"
    PRESSED = "facepressed"
    OOH = "faceooh"

    def __repr__(self) -> str:
        return str([i.value for i in GameState].index(self.value))

    def __str__(self) -> str:
        return f"{self.name}"


class CellState(Enum):
    CELL_0 = "square open0"
    CELL_1 = "square open1"
    CELL_2 = "square open2"
    CELL_3 = "square open3"
    CELL_4 = "square open4"
    CELL_5 = "square open5"
    CELL_6 = "square open6"
    CELL_7 = "square open7"
    CELL_8 = "square open8"
    UNOPENED = "square blank"
    MINE = "square bombrevealed"
    FLAG = "square bombflagged"
    BOMBDEATH = "square bombdeath"
    WALL = "wall"

    def __repr__(self) -> str:
        """"""
        if self == CellState.MINE:
            s = "B"
        elif self == CellState.FLAG:
            s = "F"
        else:
            s = str(self.num())
        return s

    def __str__(self) -> str:
        return f"{self.name}"

    def num(self) -> int:
        return [i.value for i in CellState].index(self.value)


class MineField:
    def __init__(self, headless: bool, width: int = 0, height: int = 0, n_mines: int = 0, state: str = ""):
        """Opens the game and setup the grid

        Args:
            headless: Whether to use headless mode
            width: Width of the minefield
            height: Height of the minefield
            n_mines: Number of mines
            state: A gamestate that can be imported
        """

        self._state = state

        params = (width, height, n_mines)
        if not all(params) and any(params):
            raise ValueError("Either give all the width, height, and n_mines parameters, or none at all.")
        if all(params) and state:
            raise ValueError("Either give all the grid specifications or a grid state, not both.")
        elif all(params) and bool([1 for x in params if x < 0]):
            raise ValueError("Params width, height, and n_mines must all be non negative")

        geckodriver_path = "/snap/bin/geckodriver"

        driver_service = webdriver.FirefoxService(executable_path=geckodriver_path)

        options = webdriver.FirefoxOptions()
        if headless:
            options.add_argument("-headless")

        # Setup driver and open the website
        self._driver = webdriver.Firefox(service=driver_service, options=options)  # This raises if no driver found
        wait = WebDriverWait(self._driver, 10)
        self._driver.implicitly_wait(10)
        self._driver.get("https://minesweeperonline.com/#200-night")

        # Close the cookie popup
        iframe = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "iframe#sp_message_iframe_1342217")))
        self._driver.switch_to.frame(iframe)
        accept_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[title='Accept']")))
        accept_button.click()
        self._driver.switch_to.default_content()

        # Setup the desired minefield

        if state:
            self._import_gamestate()
        elif (width, height, n_mines) != (0, 0, 0):
            if not headless:
                print("creating custom minefield...")
            self._mod_game_specs(width, height, n_mines)
        if not headless:
            print("Loading every cell into memory...")
        out = self._init_js()

        self.grid = out["grid"]
        self.height = out["rows"]
        self.width = out["cols"]

        self._face = self._driver.find_element(By.ID, "face")
        self._mine_displ = [
            self._driver.find_element(By.ID, "mines_hundreds"),
            self._driver.find_element(By.ID, "mines_tens"),
            self._driver.find_element(By.ID, "mines_ones"),
        ]
        self._time_displ = [
            self._driver.find_element(By.ID, "seconds_hundreds"),
            self._driver.find_element(By.ID, "seconds_tens"),
            self._driver.find_element(By.ID, "seconds_ones"),
        ]

        self._n_mines = self.get_mines()

        if not headless:
            print("\n============= SETUP COMPLETE =============", end="\n\n")

    def _init_js(self):
        """Determines the size of the grid and provides a function to fetch the grid"""

        s = f"""
            window.getSquareValue = function(id) {{
                const square = document.getElementById(id);
                if (!square) return null;
                if (square.style.display === 'none') return 'wall';
                return square.className;
            }}

            // determine grid size dynamically
            let rows = 0;
            let cols = 0;

            // find number of rows
            while (true) {{
                rows++;
                const cell = document.getElementById(`${{rows}}_1`);
                if (!cell || cell.style.display === 'none') {{
                    rows--;
                    break;
                }}
            }}

            // find number of columns
            while (true) {{
                cols++;
                const cell = document.getElementById(`1_${{cols}}`);
                if (!cell || cell.style.display === 'none') {{
                    cols--;
                    break;
                }}
            }}

            window.getGrid = function(rows, cols) {{
                const grid = [];
                for (let r = 1; r <= rows; r++) {{
                    const row = [];
                    for (let c = 1; c <= cols; c++) {{
                        const id = `${{r}}_${{c}}`;
                        row.push(window.getSquareValue(id));
                    }}
                    grid.push(row);
                }}
                return grid;
            }}

            // return grid along with dimensions
            return {{ grid: window.getGrid(rows, cols), rows: rows, cols: cols }};
            """
        return self._driver.execute_script(s)

    def _import_gamestate(self):
        s = f"""
        document.querySelector("textarea").value = "{self._state}" 
        document.querySelector('input[type="submit"][value="Load Game"]').click();
        """
        self._driver.execute_script(s)

    def _mod_game_specs(self, width: int, height: int, n_mines: int):
        s = f"""
        document.getElementById("custom").click();
        document.getElementById("custom_height").value = {height};
        document.getElementById("custom_width").value = {width};
        document.getElementById("custom_mines").value = {n_mines};
        document.querySelector('input[type="submit"][value="New Game"]').click();
        """
        self._driver.execute_script(s)

    def get_grid(self):
        """Returns the grid as an iterator"""
        s = f"return window.getGrid({self.height}, {self.width})"
        grid = self._driver.execute_script(s)
        return [[CellState(c).num() for c in row] for row in grid]

    def _mouse_click_on_cells(self, arr, button: int):
        js_ids = ", ".join(f'"{y + 1}_{x + 1}"' for x, y in arr)

        s = f"""
        const squareIds = [{js_ids}];
        squareIds.forEach(id => {{
            const square = document.getElementById(id);
            if (square) {{
                // simulate left click
                square.dispatchEvent(new MouseEvent("mousedown", {{ button: {button}, bubbles: true }}));
                square.dispatchEvent(new MouseEvent("mouseup", {{ button: {button}, bubbles: true }}));
            }}
        }});
        """
        self._driver.execute_script(s)

    def open_cell(self, x: int, y: int):
        """Indexing starting from 0"""
        self._mouse_click_on_cells([(x, y)], 0)

    def flag_cell(self, x: int, y: int):
        """Indexing starting from 0"""
        self._mouse_click_on_cells([(x, y)], 2)

    def get_time(self):
        """Gets the elapsed game time"""
        return int("".join([str(x.get_attribute("class")).removeprefix("time") for x in self._time_displ]))

    def get_mines(self):
        """Gets the mine amount"""
        return int("".join([str(x.get_attribute("class")).removeprefix("time") for x in self._mine_displ]))

    def get_gamestate(self):
        """Returns the gamestate"""
        return GameState(str(self._face.get_attribute("class")))

    def get_cell_state(self, x: int, y: int):
        """Returns the cells state at the given coordinates"""
        cell_str = self._driver.execute_script(f'return window.getSquareValue("{y + 1}_{x + 1}");')
        return CellState(cell_str)

    def end_session(self):
        """Exits the web driver"""
        self._driver.quit()

    def restart(self):
        """Starts a new game"""
        self._face.click()

    def print_grid(self):
        print()
        grid = self.get_grid()
        for row in grid:
            print(row)
        print()
