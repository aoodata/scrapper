from utils import *
import time

action_sleep_time = 2

class Mover:
    def __init__(self):
        pass

    def move(self, frame, screenshot):
        raise NotImplementedError

class MoverClick(Mover):
    def __init__(self, exit_name):
        super().__init__()
        self.exit_name = exit_name

    def move(self, frame, screenshot):

        exit_box = frame.boxes[self.exit_name]
        click_center(screenshot, exit_box)
        return True

class MoverKeyboard(Mover):
    def __init__(self, key="{ESC}"):
        super().__init__()
        self.key = key

    def move(self, frame, screenshot):
        send_key(self.key)
        return True


class Frame:
    def __init__(self, name, ref_box):
        self.name = name
        self.exits = {}
        image_file = r"patterns/Navigator/frame_" + name + ".png"
        pfile = r"patterns/Navigator/frame_" + name + ".pickle"
        with open(pfile, 'rb') as f:
            data = pickle.load(f)
        self.boxes = data['rectangles']
        self.ref_screenshot = cv2.imread(image_file)
        self.ref_box = self.boxes[ref_box]
        self.ref_pattern = get_box(self.ref_screenshot, self.ref_box)

    def add_exit(self, exit_name, exit_frame, mover):
        self.exits[exit_name] = (exit_frame, mover)

    def am_i_here(self, screenshot):
        """ screenshot should be opencv BGR image """
        center = find_max_pattern(screenshot, self.ref_pattern)
        if center is None:
            return False
        # check if the center is in the ref box
        if not is_in_box(screenshot, center, self.ref_box):
            return False
        return True

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


frame_main_city = Frame("main_city", "event_menu")
frame_exit = Frame("exit", "yes")
frame_my_info = Frame("my_info", "ranking")
frame_ranking_globe = Frame("ranking_globe", "globe_ranking")
frame_ranking_nation = Frame("ranking_nation", "nation_ranking")
frame_event_menu = Frame("event_menu", "event_menu")

frame_main_city.add_exit("exit", frame_exit, MoverKeyboard())
frame_exit.add_exit("main_city", frame_main_city, MoverKeyboard())
frame_main_city.add_exit("my_info", frame_my_info, MoverClick("my_info"))
frame_main_city.add_exit("event_menu", frame_event_menu, MoverClick("event_menu"))
frame_event_menu.add_exit("main_city", frame_main_city, MoverKeyboard())
frame_my_info.add_exit("ranking_globe", frame_ranking_globe, MoverClick("ranking"))
frame_my_info.add_exit("main_city", frame_main_city, MoverKeyboard())
frame_ranking_globe.add_exit("ranking_nation", frame_ranking_nation, MoverClick("nation_ranking"))
frame_ranking_globe.add_exit("my_info", frame_my_info, MoverKeyboard())
frame_ranking_nation.add_exit("ranking_globe", frame_ranking_globe, MoverClick("globe_ranking"))
frame_ranking_nation.add_exit("my_info", frame_my_info, MoverKeyboard())

class Navigator:
    def __init__(self, window):
        self.frames = {}
        for frame in [frame_main_city, frame_exit, frame_my_info, frame_ranking_globe, frame_ranking_nation, frame_event_menu]:
            self.frames[frame.name] = frame
        self.window = window
        self.start_frame = frame_main_city
        self.current_frame = frame_main_city

    def find_shortest_path(self, start_frame, end_frame):
        """ find the shortest path from start_frame to end_frame """
        visited = set()
        visited.add(start_frame)
        queue = [(start_frame, [])]
        while queue:
            frame, path = queue.pop(0)
            if frame.name == end_frame.name:
                return path
            for _, (exit_frame, _) in frame.exits.items():
                if exit_frame not in visited:
                    visited.add(exit_frame)
                    queue.append((exit_frame, path + [exit_frame]))
        return None

    def reset(self):
        time.sleep(action_sleep_time)
        image = capture_window(self.window)
        counter = 0
        while not self.start_frame.am_i_here(image):
            send_key("{ESC}")
            time.sleep(action_sleep_time)
            image = capture_window(self.window)
            counter += 1
            if counter > 10:
                raise Exception("Cannot find the start frame")
        self.current_frame = self.start_frame

    def relocate(self):
        time.sleep(action_sleep_time)
        image = capture_window(self.window)
        screenshot = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for frame in self.frames.values():
            if frame.am_i_here(screenshot):
                self.current_frame = frame
                print("You are here: ", frame.name)
                return True

        print("Cannot find the current frame")
        return False

    def navigate(self, end_frame_name):
        end_frame = self.frames[end_frame_name]
        time.sleep(action_sleep_time)
        path = self.find_shortest_path(self.current_frame, end_frame)
        if path is None:
            return False
        for frame in path:
            _, mover = self.current_frame.exits[frame.name]
            image = capture_window(self.window)
            screenshot = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.window.set_focus()
            if not mover.move(self.current_frame, screenshot):
                return False
            time.sleep(action_sleep_time)
            self.current_frame = frame
        return True



if __name__ == "__main__":
    pass