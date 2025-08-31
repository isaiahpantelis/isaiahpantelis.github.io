---
layout: default
title: Programming
permalink: /programming
---

# C#
## Encapsulation
### Properties

Encapsulation is a design principle in object-oriented programming (OOP) that promotes techniques for protecting the state of an object from corruption. One design pattern that supports encapsulation and is as old as OOP itself is the implementation of so-called `setter` and `getter` methods. In C#, `properties` are the built-in language support for this idea and they are very similar with python properties. The main advantages of using properties are:

- encapsulation
- data validation
- code reuse in constructors
- less visual clutter at the point of use

A class that uses public fields:

```csharp
class Instrument
{
    public int ID;  // -- Using int for compatibility with CLS and various APIs. But should be non-negative.
    public string CUSIP = "";  // -- Has to be of length 8
    public string ISIN = "";  // -- Has to be of length 11
    public string PARSEKEYABLE = "";  // -- Has to end in "\tCode"

    // -- Override ToString() method to print out a summary of the instrument.
    public override string ToString()
    {
        return $"Instrument(ID={ID}, CUSIP={CUSIP}, ISIN={ISIN}, PARSEKEYABLE={PARSEKEYABLE})";
    }
}
```

A class that uses properties to encapsulate its fields:

```csharp
class Product
{
    private int _ID;
    private string _CUSIP = "";
    private string _ISIN = "";
    private string _PARSEKEYABLE = "";

    // -- And an automatic property -- //
    public string Blob { get; set; }
    
    public int ID
    {
        get => _ID;
        set
        {
            if (value < 0)
            {
                throw new ArgumentException("ID must be non-negative.");
            }
            _ID = value;
        }
    }

    public string CUSIP
    {
        get => _CUSIP;
        set
        {
            if (value.Length != 8)
            {
                throw new ArgumentException("CUSIP must be 8 characters long.");
            }
            _CUSIP = value;
        }
    }

    public string ISIN
    {
        get => _ISIN;
        set
        {
            if (value.Length != 11)
            {
                throw new ArgumentException("ISIN must be 11 characters long.");
            }
            _ISIN = value;
        }
    }

    public string PARSEKEYABLE
    {
        get => _PARSEKEYABLE;
        set
        {
            if (!value.EndsWith("\tCode"))
            {
                throw new ArgumentException("PARSEKEYABLE must end with '\\tCode'.");
            }
            _PARSEKEYABLE = value;
        }
    }

    public Product(int id, string cusip, string isin, string parsekeyable, string blob)
    {
        ID = id;
        CUSIP = cusip;
        ISIN = isin;
        PARSEKEYABLE = parsekeyable;
        Blob = blob;
    }

    public override string ToString()
    {
        return $"Product(ID={ID}, CUSIP={CUSIP}, ISIN={ISIN}, PARSEKEYABLE={PARSEKEYABLE},  Blob={Blob})";
    }
}
```

Public attributes are the Wild West. Properties provide extra sweet syntactic sugar (in addition to the benefits listed above):

```csharp
class Program
{

    static void Main(string[] args)
    {
        
        // -- Create a new instrument. Set fields to illegal values. Anything goes.
        var bond = new Instrument { ID = -1, CUSIP = "say", ISIN = "what", PARSEKEYABLE = "now?" };
        Console.WriteLine(bond.ToString());
        
        // -- All commented out definitions will throw an exception.
        // var loan = new Product {ID = -1, CUSIP = "say", ISIN = "what", PARSEKEYABLE = "now?"};
        // var loan = new Product {ID = 42, CUSIP = "say", ISIN = "what", PARSEKEYABLE = "now?"};
        // var loan = new Product {ID = 42, CUSIP = "XS123456", ISIN = "what", PARSEKEYABLE = "now?"};
        // -- This definition meets all the constraints:
        var loan = new Product(42, "XS123456", "XS123456789", "B12345\tCode", "Lorem ipsum");
        Console.WriteLine(loan.ToString());
        // -- The beauty of properties: concise method invocation, as if accessing a public attribute.
        loan.CUSIP += "7";  // -- This will throw an exception because `CUSIP` has to be 8-characters long.
    }
    
}
```