FINE_AMOUNT = 25

def issue_ticket(speed: float):
    """Stores tickets issued in Speeding_Ticket.txt
    
    Args:
        speed: float obtained through estimate_speed()

    Returns:
        A txt file named Speeding_Ticket.txt with detected speeds and fine amounts
    """
    filename = "Speeding_Ticket.txt"
    with open (filename,"a") as file:
        file.write("Speeding Ticket\n")
        file.write(f"Detected Speed: {speed:.2f} km/h\n")
        file.write(f"Fine Amount: ${FINE_AMOUNT}\n")
        file.write("\n")