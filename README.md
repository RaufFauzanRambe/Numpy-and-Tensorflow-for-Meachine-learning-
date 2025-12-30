from datetime import datetime
import time

main = input("Enter your game: ")

if main == "Snakezone":
  print("Game:", main)
elif main == "zuma":
  print("Game:", main)
elif main == "Train to city":
  print("Game:",main)
elif main == "Star the leader":
  print("Game:", main)
else:
  print("This have choice the 5 game")

now = datetime.time().strftime("%H:%M:%S | Date: %d:%m:Y")
print("ON by the system:", now)

system_type_controller = ("Controller:", "X, Y, W, O")
system_zoom_level = ("Zoom Level:", ("9, 13, 45, 321"))
system_on_off_state = ("state:",[93, 23, 123, 54])

if system_type_controller == "Mechanical controller X, Y, O, W":
  print("System control state: True (Mechanical controller)")
elif system_on_off_state == "C4, X1, Full trained": # Changed => to ==
  print("System on/off state: False (Fully trained)")
elif system_zoom_level == "Quadratic, system zoom": # Changed <= to ==
  print("System zoom level: True (Quadratic system zoom)")
else:
  print("Default system state: True")

actual_system_on_off_status = "ON" # Example: can be "ON" or "OFF"

print(f"Started by system: {actual_system_on_off_status}")
print(f"You like the game: {main}")
