import numpy as np

def check_state_evolution_encoding(s, dimensions, dt = 0.2):
    """
        Verify that variables in s are encoded correctly
    """
    
    Tx = dimensions.index('taker_x')
    Tdx = dimensions.index('taker_dx')
    Tddx = dimensions.index('taker_ddx')

    Gx = dimensions.index('giver_x')
    Gdx = dimensions.index('giver_dx')
    Gddx = dimensions.index('giver_ddx')

    Relx = dimensions.index('relative_x')
    Reldx = dimensions.index('relative_dx')
    Relddx = dimensions.index('relative_ddx')
    
    for t in range(0, len(s)-2):
        for i in [0,1,2]:
            G = Gx+i
            Gd = Gdx+i
            Gdd = Gddx+i
            T = Tx+i
            Td = Tdx+i
            Tdd = Tddx+i
            Rel = Relx+i
            Reld = Reldx+i
            Reldd = Relddx+i
            try: # Check acceleration encoding
                G_encoded_acc = s[t][Gdd].varValue # Giver
                G_calculated_acc = (s[t][G].varValue - 2* s[t-1][G].varValue + s[t-2][G].varValue) / dt**2
                if not( G_encoded_acc == G_calculated_acc ):
                    G_acc_error = G_encoded_acc - G_calculated_acc
                    if G_acc_error > 1e-10:
                        #print("Acceleration encoding error!")
                        print("Acceleration encoding error! \n error: " + str(G_acc_error))
                        print(" planning step" + str(t))
            except:
                pass
            try:  
                T_encoded_acc = s[t][Tdd].varValue # Taker
                T_calculated_acc = (s[t][T].varValue - 2* s[t-1][T].varValue + s[t-2][T].varValue) / dt**2
                if not( T_encoded_acc == T_calculated_acc ):
                    T_acc_error = T_encoded_acc - T_calculated_acc
                    if T_acc_error > 1e-10:
                        #print("Acceleration encoding error!")
                        print("Acceleration encoding error! \n error: " + str(T_acc_error))
                        print(" planning step" + str(t))
            except:
                pass
            try: # Check velocity encoding
                G_encoded_vel = s[t][Gd].varValue
                G_calculated_vel = (s[t][G].varValue - s[t-1][G].varValue) / dt
                if not( G_encoded_vel == G_calculated_vel ):
                    G_vel_error = G_encoded_vel - G_calculated_vel
                    if G_vel_error > 1e-10:
                        #print("Velocity encoding error!")
                        print("Velocity encoding error!\n error: " + str(G_vel_error))
                        print(" planning step" + str(t))
            except:
                pass    
            try: # Check velocity encoding
                T_encoded_vel = s[t][Td].varValue
                T_calculated_vel = (s[t][T].varValue - s[t-1][T].varValue) / dt
                if not( T_encoded_vel == T_calculated_vel ):
                    T_vel_error = T_encoded_vel - T_calculated_vel
                    if T_vel_error > 1e-10:
                        #print("Velocity encoding error!")
                        print("Velocity encoding error!\n error: " + str(T_vel_error))
                        print(" planning step" + str(t))
            except:
                pass    
            try: # Check relative position encoding
                Rel_encoded_pos = s[t][Rel].varValue
                Rel_calculated_pos = (s[t][G].varValue - s[t][T].varValue)
                if not( Rel_encoded_pos == Rel_calculated_pos ):
                    Rel_pos_error = Rel_encoded_pos - Rel_calculated_pos
                    if Rel_pos_error > 1e-10:
                        #print("Velocity encoding error!")
                        print("Relative position encoding error!\n error: " + str(Rel_pos_error ))
                        print(" planning step" + str(t))
            except:
                pass       
            try: # Check relative velocity encoding
                Rel_encoded_vel = s[t][Reld].varValue
                Rel_calculated_vel = (s[t][Gd].varValue - s[t][Td].varValue)
                if not( Rel_encoded_vel == Rel_calculated_vel ):
                    Rel_vel_error = Rel_encoded_vel - Rel_calculated_vel
                    if Rel_vel_error > 1e-10:
                        #print("Velocity encoding error!")
                        print("Relative velocity encoding error!\n error: " + str(Rel_vel_error ))
                        print(" planning step" + str(t))
            except:
                pass       
            try: # Check relative acceleration encoding
                Rel_encoded_acc = s[t][Reldd].varValue
                Rel_calculated_acc = (s[t][Gdd].varValue - s[t][Tdd].varValue)
                if not( Rel_encoded_acc == Rel_calculated_acc ):
                    Rel_acc_error = Rel_encoded_acc - Rel_calculated_acc
                    if Rel_acc_error > 1e-10:
                        #print("Velocity encoding error!")
                        print("Relative acceleration encoding error!\n error: " + str(Rel_acc_error ))
                        print(" planning step" + str(t))
            except:
                pass       


