float32 q            # motor current position (rad)
float32 dq           # motor current speed (rad/s)
float32 ddq          # motor current speed (rad/s)
float32 tau_est      # current estimated output torque (N*m)
float32 cur          # current estimated output cur (N*m)
float32 tau_cmd   # desired (commanded) joint torque (N·m)
uint16 status_word
uint16  cmd_mode
float32 cur_cmd