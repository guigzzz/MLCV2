from queue import queue

class Lift:
    def __init__(self):
        self.current_floor = 0
        self.end_floor = 0

        self.floors_to_visit = sortedDatastructure()

        self.direction = None


    def add_destination(self, floor):
        self.floors_to_visit.insert(floor)

        if self.floors_to_visit.empty():
            if floor < self.current_floor:
                self.direction = 'down'
            else:
                self.direction = 'up'

    def get_next_destination(self):
        next_floor = self.floors_to_visit.pop()
        self.current_floor = next_floor


class LiftScheduler:
    def __init__(self, number_of_lifts, number_of_floors):
        self.number_of_floors = number_of_floors
        self.number_of_lifts = number_of_lifts


        self.lifts = [Lift() for i in range(self.number_of_lifts)]

        self.is_waiting_on_floor = [
            False for _ in range(self.number_of_floors)
        ]

        # self.idle_lift_queue = queue()
        # [self.idle_lift_queue.put(l) for l in self.lifts]

        # self.non_idle_lift_queue = queue()


    def request_lift(self, floor):
        # if not self.idle_lift_queue.empty():
        #     lift = self.idle_lift_queue.get()

        closest_lift_floors = 99999
        closest_lift = None

        for lift in self.lifts:
            if floor - lift.current_floor < closest_lift_floors and lift.direction == 'down':
                closest_lift_floors = floor - lift.current_floor
                closest_lift = lift

            elif lift.current_floor - floor < closest_lift_floors and lift.direction == 'up':
                closest_lift_floors = lift.current_floor - floor

    


            self.non_idle_lift_queue.put(lift)
            lift.go_to_floor(end_floor)


    def 

        else:
            self.is_waiting_on_floor[current_floor] = True


    

            

    
