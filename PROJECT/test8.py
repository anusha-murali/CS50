from prettytable import PrettyTable


table = PrettyTable(['Coin', 'Price', 'High', 'Low'])

table.add_row(['BTC', '14525.00 USD', '15355.00 USD', '13755.00 USD'])
table.add_row(['ETH', '1191.00 USD', '1250.00 USD', '965.18 USD'])
table.add_row(['XRP', '2.25 USD', '2.49 USD', '1.90 USD'])
table.add_row(['LTC', '247.72 USD', '258.04 USD', '230.18 USD'])
table.add_row(['MIOTA', '3.64 USD', '3.95 USD', '3.15 USD'])


print(table)
