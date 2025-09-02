---
layout: default
title: Programming
permalink: /programming
---

# ToC
- [C#](#c)
  - [Encapsulation](#encapsulation)
    - [Properties](#properties)
  - [Core language](#core-language)
    - [Nullable types](#nullable-types)

<hr>

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
    public string CUSIP = "";  // -- Has to be of length 9
    public string ISIN = "";  // -- Has to be of length 12
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
            if (value.Length != 9)
            {
                throw new ArgumentException("CUSIP must be 9 characters long.");
            }
            _CUSIP = value;
        }
    }

    public string ISIN
    {
        get => _ISIN;
        set
        {
            if (value.Length != 12)
            {
                throw new ArgumentException("ISIN must be 12 characters long.");
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
        loan.CUSIP += "7";  // -- This will throw an exception because `CUSIP` has to be 9-characters long.
    }
    
}
```

Output:

```
Instrument(ID=-1, CUSIP=say, ISIN=what, PARSEKEYABLE=now?)
Product(ID=42, CUSIP=XS123456, ISIN=XS123456789, PARSEKEYABLE=B12345    Code,  Blob=Lorem ipsum)
Unhandled exception. System.ArgumentException: CUSIP must be 9 characters long.
   at Properties.Program.Product.set_CUSIP(String value) in /Users/pdawg/Projects/CSharp/BabySteps/Properties/Program.cs:line 64
   at Properties.Program.Main(String[] args) in /Users/pdawg/Projects/CSharp/BabySteps/Properties/Program.cs:line 129
```

<br>

## Core language
### Nullable types
In a database, there can be compelling reasons for defining a field/column as `nullable`. However, a value type in C# can only assume any one of the values in the permissible range of the type; for example, an `int` can only assume one of the values in the range of `int`. To address this and many other issues, `Nullable<T>` extends the range of values of the type `T` to include the value `null`.

```csharp
class Program
{

    class DBReader
    {
        public int? ID { get; set; }
        public bool? Valid { get; set; }
    }

    static void Main(string[] args)
    {
        // -- Illegal: `id` can only take one of the values in the range of the `int` type. -- //
        // int id = null;  // -- will not compile
        
        // -- Nullable<T> is a type that extends the range of `T` to include the value `null` -- //
        Nullable<int> smart_id = null;
        
        Console.WriteLine(smart_id == null);
        Console.WriteLine(smart_id.HasValue);
        
        // -- More compact syntax -- //
        int? smarter_id = null;
        
        Console.WriteLine(smarter_id == null);
        Console.WriteLine(smarter_id.HasValue);
        
        // -- Of course, it works with arrays too. -- //
        int?[] arrayOfNullableInts = new int?[10];
        
        foreach (var item in arrayOfNullableInts)
        {
            Console.WriteLine(item ?? 0);  // -- null-coallescing operator
        }
        
        // -- Assignment -- //
        int? initially_null_int = null;
        initially_null_int ??= 42;
        initially_null_int ??= 667;
        
        // -- What is the output? -- //
        Console.WriteLine(initially_null_int);
        
        // -- Null-conditional + null-coallescing -- //
        Console.WriteLine($"Number of input arguments: {args?.Length ?? 0}\n"); 
       
        // -- Properties work with nullable types (imagine if it were otherwise...) -- //
        var db_reader = new DBReader();
        var id = db_reader.ID;
        var valid = db_reader.Valid;

        // -- No error: `new_id` is deduced to be of type `int?` with value `null`.
        var new_id = id + 1;

         Console.WriteLine(new_id);
    }
}
```