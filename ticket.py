FINE_AMOUNT = 25

def issue_ticket(speed):
    filename = "Speeding_Ticket.txt"
    with open (filename,"a") as file:
        file.write("Speeding Ticket\n")
        file.write(f"Detected Speed: {speed:.2f} km/h\n")
        file.write(f"Fine Amount: ${FINE_AMOUNT}\n")
        file.write("\n")